"""
physnet.py
==========
Physics-guided forecaster ("PhysNet") for the Bleaching Alert level 1-14 days ahead.

NOAA's Degree Heating Weeks follow a fixed formula:
    DHW(t) = (1/7) * sum of HotSpot values >= 1 degC over the 84 days ending on t
so, for a lead of k days,
    DHW(t+k) = DHW(t) + (1/7) * sum_{j=1..k} HS+(t+j)  -  (1/7) * sum_{j=1..k} HS+(t-84+j)
               known        ^ must be forecast            ^ known from history ("drop-off")
where HS+ = HotSpot if >= 1 else 0. The alert level then follows from HotSpot and DHW.

So the network only has to forecast HotSpot for the next 14 days; a formula layer turns
that into DHW exactly as NOAA does, and a final layer outputs the alert level for each
lead, seeing both the learned features and the formula's result.

Inputs for issue day t (all scaled to roughly unit range):
    10 frames = the 7 days t-6..t  +  weekly means of the 3 weeks before (4 weeks of history)
    each frame: SST anomaly, HotSpot, DHW, SST [+ wind, cloud, radiation, air temperature]
                + mask, latitude, longitude, season [+ ENSO, MJO]
    head extras: DHW(t), the known drop-off for leads 1..14 [+ weather means for the next
                 two weeks - from forecasts when run live]

Training loss: cross-entropy per lead (mildly class-weighted) + an ordinal term
(levels are ordered) + Huber loss on the HotSpot forecast.
"""

import datetime as dt
import json
from dataclasses import asdict, dataclass

import numpy as np

LEADS = 14
N_CLASSES = 5
DAILY = 7
WEEKS = 3


@dataclass
class Config:
    physics: bool = True           # HotSpot head + DHW formula layer
    weather_past: bool = True      # observed weather in the input frames
    weather_future: bool = True    # weather for the next two weeks (forecast when live)
    indices: bool = True           # ENSO + MJO
    future_air_temp: bool = True   # include air temperature in the future weather (ERA5 air temp over sea follows observed SST)
    target: str = "daily"          # "daily" alert level, or "max7" = NOAA's 7-day maximum alert
    forecast_noise: bool = False   # add realistic weather-forecast error to the future weather in training
    hidden: int = 32
    epochs: int = 30
    lr: float = 2e-3
    batch: int = 8
    patience: int = 7
    seed: int = 0
    name: str = "PhysNet"


# ── Weather on the forecast grid ──────────────────────────────────────
class Drivers:
    """Weather (interpolated to the 0.1 deg grid on the fly) and climate indices by date."""

    def __init__(self, path, lats, lons):
        d = np.load(path)
        self._setup([dt.date.fromisoformat(s) for s in d["dates"]], d["points"], d["vals"], d["nino34"], d["mjo"], lats, lons)

    @classmethod
    def from_arrays(cls, dates, points, vals, nino, mjo, lats, lons):
        obj = cls.__new__(cls)
        obj._setup(list(dates), np.asarray(points), np.asarray(vals, np.float32), np.asarray(nino, np.float32),
                   np.asarray(mjo, np.float32), lats, lons)
        return obj

    def _setup(self, dates, pts, vals, nino, mjo, lats, lons):
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        self.dates = dates
        self.index = {day: i for i, day in enumerate(self.dates)}
        self.vals = vals                           # [T, P, 4]
        self.nino, self.mjo = nino, mjo
        tgt = np.stack(np.meshgrid(lats, lons, indexing="ij"), -1).reshape(-1, 2)
        # interpolation matrix: grid = M @ point_values (linear inside the hull, nearest outside)
        m = np.zeros((len(tgt), len(pts)), np.float32)
        for p in range(len(pts)):
            e = np.zeros(len(pts)); e[p] = 1
            lin = LinearNDInterpolator(pts, e)(tgt)
            near = NearestNDInterpolator(pts, e)(tgt)
            m[:, p] = np.where(np.isnan(lin), near, lin)
        self.M = m
        self.shape = (len(lats), len(lons))
        # fill gaps in point series by time interpolation, then scale
        v = self.vals.copy()
        for p in range(v.shape[1]):
            for k in range(v.shape[2]):
                s = v[:, p, k]; bad = np.isnan(s)
                if bad.any() and (~bad).any():
                    s[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), s[~bad])
        self.scaled = ((v - np.array([20, 50, 20, 27], np.float32)) / np.array([10, 40, 8, 3], np.float32)).astype(np.float32)

    def grid(self, days):
        """Mean scaled weather over `days` -> [4, H, W] (zeros if unavailable)."""
        idx = [self.index[d] for d in days if d in self.index]
        if not idx:
            return np.zeros((4,) + self.shape, np.float32)
        pv = self.scaled[idx].mean(0)                            # [P, 4]
        return (self.M @ pv).T.reshape((4,) + self.shape)

    def scalars(self, day):
        i = self.index.get(day)
        if i is None:
            return np.zeros(3, np.float32)
        return np.array([self.nino[i] / 2, self.mjo[i, 0] / 2, self.mjo[i, 1] / 2], np.float32)


# ── Sample construction ───────────────────────────────────────────────
OCEAN_SCALE = np.array([2.0, 2.0, 8.0, 2.0], np.float32)
OCEAN_SHIFT = np.array([0.0, 0.0, 0.0, 28.0], np.float32)


def season_phase(day):
    start = dt.date(day.year if day.month >= 7 else day.year - 1, 11, 1)
    a = (day - start).days / 365.25 * 2 * np.pi
    return np.sin(a), np.cos(a)


class Builder:
    """Builds network inputs and targets for issue day t from a Cube (+ Drivers)."""

    def __init__(self, cube, drivers, cfg):
        self.c, self.d, self.cfg = cube, drivers, cfg
        H, W = cube.valid.shape
        self.pad = (H + H % 2, W + W % 2)
        lat2, lon2 = np.meshgrid(cube.lats, cube.lons, indexing="ij")
        self.latc = ((lat2 + 15.7) / 3).astype(np.float32)
        self.lonc = ((lon2 - 147.6) / 3).astype(np.float32)
        # Precomputed once, stored for Marine Park cells only (1,927 of 11,342) to keep memory low:
        # scaled ocean channels, their 7-day means ending on each day, and HotSpots >= 1.
        v = cube.valid
        xc = cube.x[:, :4][:, :, v].astype(np.float32)                                  # [T, 4, N]
        hs = xc[:, 1]
        self.hs_plus = np.nan_to_num(np.where(hs >= 1, hs, 0)).astype(np.float16)     # [T, N]
        ocean = (xc - OCEAN_SHIFT[None, :, None]) / OCEAN_SCALE[None, :, None]
        del xc
        self.ocean = np.nan_to_num(ocean).astype(np.float16)
        wk = np.zeros_like(self.ocean)
        for i, day in enumerate(cube.dates):
            ids = [cube.index[day - dt.timedelta(days=k)] for k in range(7) if day - dt.timedelta(days=k) in cube.index]
            with np.errstate(all="ignore"):
                wk[i] = np.nanmean(ocean[ids], 0) if len(ids) > 1 else ocean[ids[0]]
        self.ocean_wk = np.nan_to_num(wk).astype(np.float16)
        del ocean, wk

    def _full(self, compact, channels=None):
        """Scatter park-cell values back onto the full grid (zeros elsewhere)."""
        H, W = self.c.valid.shape
        out = np.zeros(((channels,) if channels else ()) + (H, W), np.float32)
        out[..., self.c.valid] = compact
        return out

    def n_frame_channels(self):
        return 4 + (4 if self.cfg.weather_past else 0) + 5 + (3 if self.cfg.indices else 0)

    def n_extra_channels(self):
        return 1 + LEADS + (8 if self.cfg.weather_future else 0) + (LEADS if self.cfg.target == "max7" else 0)

    def known_max(self, t):
        """For the 7-day-maximum target: the highest level already observed inside each lead's
        window (days t+k-6 .. t), or -1 when no part of the window has happened yet."""
        day = self.c.dates[t]
        H, W = self.c.valid.shape
        out = np.full((LEADS, H, W), -1.0, np.float32)
        for k in range(1, 7):
            ids = [self.c.index[day - dt.timedelta(days=j)] for j in range(0, 7 - k) if day - dt.timedelta(days=j) in self.c.index]
            if ids:
                with np.errstate(all="ignore"):
                    out[k - 1] = np.nan_to_num(np.nanmax(self.c.baa[ids], axis=0), nan=-1.0)
        return out

    def _ocean(self, days):
        """Scaled ocean channels: one day, or a 7-day block (uses the precomputed 7-day mean)."""
        end = days[0] if len(days) == 1 else days[0]      # weekly groups list their end day first
        i = self.c.index.get(end)
        if i is None:
            return np.zeros((4,) + self.c.valid.shape, np.float32)
        return self._full((self.ocean if len(days) == 1 else self.ocean_wk)[i].astype(np.float32), 4)

    def frames(self, t):
        day = self.c.dates[t]
        groups = [[day - dt.timedelta(days=k)] for k in range(DAILY - 1, -1, -1)]
        weekly = [[day - dt.timedelta(days=DAILY + 7 * w + k) for k in range(7)] for w in range(WEEKS - 1, -1, -1)]
        groups = weekly + groups                                   # oldest first
        H, W = self.c.valid.shape
        s, co = season_phase(day)
        base = [self.c.valid.astype(np.float32), self.latc, self.lonc,
                np.full((H, W), s, np.float32), np.full((H, W), co, np.float32)]
        out = []
        for g in groups:
            ch = [self._ocean(g)]
            if self.cfg.weather_past:
                ch.append(self.d.grid(g))
            ch.append(np.stack(base))
            if self.cfg.indices:
                sc = self.d.scalars(g[-1])
                ch.append(np.broadcast_to(sc[:, None, None], (3, H, W)).astype(np.float32))
            out.append(np.concatenate(ch))
        return np.stack(out)                                       # [10, C, H, W]

    def dropoff(self, t):
        """Known DHW drop-off for leads 1..LEADS: cumulative HS+ of days t-84+1 .. t-84+k, / 7."""
        day = self.c.dates[t]
        vals = []
        for j in range(1, LEADS + 1):
            q = day - dt.timedelta(days=84 - j)
            i = self.c.index.get(q)
            vals.append(self._full(self.hs_plus[i].astype(np.float32)) if i is not None else np.zeros(self.c.valid.shape, np.float32))
        return np.cumsum(np.stack(vals), 0) / 7.0                  # [LEADS, H, W]

    def extras(self, t, future_weather=None):
        day = self.c.dates[t]
        dhw_now = np.nan_to_num(self.c.x[t, 2].astype(np.float32))
        ch = [dhw_now[None] / 8.0, self.dropoff(t) / 8.0]
        if self.cfg.weather_future:
            if future_weather is None:
                wk1 = self.d.grid([day + dt.timedelta(days=k) for k in range(1, 8)])
                wk2 = self.d.grid([day + dt.timedelta(days=k) for k in range(8, 15)])
                future_weather = np.concatenate([wk1, wk2])
            if not self.cfg.future_air_temp:
                future_weather = future_weather.copy()
                future_weather[[3, 7]] = 0.0          # blank air temperature in both weeks
            ch.append(future_weather)
        if self.cfg.target == "max7":
            ch.append(self.known_max(t) / 4.0)
        return np.concatenate(ch)                                  # [E, H, W]

    def targets(self, t):
        """HotSpot [LEADS, H, W] (NaN if missing) and level [LEADS, H, W] (-1 if missing)."""
        day = self.c.dates[t]
        hs = np.full((LEADS,) + self.c.valid.shape, np.nan, np.float32)
        lv = np.full((LEADS,) + self.c.valid.shape, -1, np.int64)
        for k in range(1, LEADS + 1):
            i = self.c.index.get(day + dt.timedelta(days=k))
            if i is not None:
                hs[k - 1] = self.c.x[i, 1]
                b = self.c.baa7[i] if self.cfg.target == "max7" else self.c.baa[i]
                lv[k - 1] = np.where(np.isnan(b), -1, b)
        hs[:, ~self.c.valid] = np.nan
        lv[:, ~self.c.valid] = -1
        return hs, lv

    def pad2(self, a):
        H, W = a.shape[-2:]
        return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, self.pad[0] - H), (0, self.pad[1] - W)])


# ── Network ───────────────────────────────────────────────────────────
def build_net(frame_ch, extra_ch, cfg):
    import torch
    import torch.nn as nn

    hid = cfg.hidden

    class ConvLSTMCell(nn.Module):
        def __init__(self):
            super().__init__()
            self.gates = nn.Conv2d(2 * hid, 4 * hid, 3, padding=1)

        def forward(self, x, h, c):
            i, f, o, g = self.gates(torch.cat([x, h], 1)).chunk(4, 1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            return torch.sigmoid(o) * torch.tanh(c), c

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Conv2d(frame_ch, hid, 3, padding=1), nn.BatchNorm2d(hid), nn.ReLU(),
                                     nn.Conv2d(hid, hid, 3, padding=1), nn.BatchNorm2d(hid), nn.ReLU())
            self.down = nn.Conv2d(hid, hid, 3, stride=2, padding=1)
            self.cell = ConvLSTMCell()
            self.up = nn.ConvTranspose2d(hid, hid, 2, stride=2)
            trunk_in = 2 * hid + extra_ch
            self.trunk = nn.Sequential(nn.Conv2d(trunk_in, 2 * hid, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.1),
                                       nn.Conv2d(2 * hid, 2 * hid, 3, padding=1), nn.ReLU())
            self.hs_head = nn.Conv2d(2 * hid, LEADS, 1) if cfg.physics else None
            head_in = 2 * hid + (4 * LEADS if cfg.physics else 0)
            self.cls_head = nn.Sequential(nn.Conv2d(head_in, 2 * hid, 1), nn.ReLU(), nn.Conv2d(2 * hid, LEADS * N_CLASSES, 1))

        def forward(self, frames, extras):
            b, t, _, h, w = frames.shape
            feats = [self.enc(frames[:, k]) for k in range(t)]
            small = [self.down(f) for f in feats]
            hs = torch.zeros_like(small[0]); cs = torch.zeros_like(small[0])
            for s in small:
                hs, cs = self.cell(s, hs, cs)
            z = self.trunk(torch.cat([self.up(hs)[:, :, :h, :w], feats[-1], extras], 1))
            hs_hat = None
            head_in = [z]
            if self.hs_head is not None:
                hs_hat = self.hs_head(z) * 2.0                                   # degC
                # --- NOAA DHW formula layer ---
                dhw_now = extras[:, :1] * 8.0
                drop = extras[:, 1:1 + LEADS] * 8.0
                hs_pos = hs_hat * torch.sigmoid((hs_hat - 1.0) / 0.1)          # soft "HotSpot >= 1"
                dhw_hat = dhw_now + torch.cumsum(hs_pos, 1) / 7.0 - drop
                head_in += [hs_hat / 2.0, (dhw_hat - 4.0) / 4.0,
                            torch.sigmoid(hs_hat / 0.1), torch.sigmoid((hs_hat - 1.0) / 0.1)]
            logits = self.cls_head(torch.cat(head_in, 1)).view(b, LEADS, N_CLASSES, h, w)
            return logits, hs_hat

    return Net()


# ── Training / inference ──────────────────────────────────────────────
class PhysNet:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history = []

    @property
    def name(self):
        return self.cfg.name

    def _device(self):
        import torch
        return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    def _batch(self, idx, with_targets=True):
        import torch
        b = self.builder
        fr = np.stack([b.pad2(b.frames(t)) for t in idx])
        fw = getattr(self, "_future", None) or {}
        ex = np.stack([b.pad2(b.extras(t, fw.get(t))) for t in idx])
        if getattr(self, "_training", False) and self.cfg.forecast_noise and self.cfg.weather_future:
            ex = self._add_forecast_error(ex)
        out = [torch.from_numpy(fr).to(self.dev), torch.from_numpy(ex).to(self.dev)]
        if with_targets:
            H, W = b.c.valid.shape
            hs_pad = np.full((len(idx), LEADS) + b.pad, np.nan, np.float32)
            lv_pad = np.full((len(idx), LEADS) + b.pad, -1, np.int64)
            for k, t in enumerate(idx):
                hs, lv = b.targets(t)
                hs_pad[k, :, :H, :W] = hs
                lv_pad[k, :, :H, :W] = lv
            out += [torch.from_numpy(hs_pad).to(self.dev), torch.from_numpy(lv_pad).to(self.dev)]
        return out

    def _add_forecast_error(self, ex):
        """Perturb the future-weather channels with realistic forecast error (training only).
        Error = bias + large-scale part shared by all points + local part, per variable and week,
        sized from real archived forecasts (forecast_error.py), then interpolated to the grid."""
        st = self._err_stats
        M, (H, W) = self.builder.d.M, self.builder.c.valid.shape
        n_pts = M.shape[1]
        base = 1 + LEADS
        for i in range(len(ex)):
            for w, key in enumerate(("week1", "week2")):
                mean, std, shared = (np.array(st[key][k], np.float32) for k in ("mean", "std", "shared"))
                common = np.random.randn(4) * std * np.sqrt(shared)
                local = np.random.randn(n_pts, 4) * std * np.sqrt(1 - shared)
                pv = mean + common + local                                   # [P, 4]
                if not self.cfg.future_air_temp:
                    pv[:, 3] = 0.0
                ex[i, base + 4 * w: base + 4 * w + 4, :H, :W] += (M @ pv).T.reshape(4, H, W)
        return ex

    def fit(self, cube, drivers, train_idx, val_idx, score_fn):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        cfg = self.cfg
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.dev = self._device()
        self.builder = Builder(cube, drivers, cfg)
        self.net = build_net(self.builder.n_frame_channels(), self.builder.n_extra_channels(), cfg).to(self.dev)

        tgt = cube.baa7 if cfg.target == "max7" else cube.baa
        counts = np.bincount(np.concatenate([np.nan_to_num(tgt[cube.index[cube.dates[t] + dt.timedelta(days=7)]][cube.valid]).astype(int)
                                             for t in train_idx]), minlength=N_CLASSES)
        w = (counts.sum() / np.maximum(counts, 1)) ** 0.25          # mild re-weighting of rare levels
        weights = torch.tensor(w / w.mean(), dtype=torch.float32, device=self.dev)
        levels = torch.arange(N_CLASSES, dtype=torch.float32, device=self.dev).view(1, 1, N_CLASSES, 1, 1)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * ((len(train_idx) + cfg.batch - 1) // cfg.batch))

        if cfg.forecast_noise:
            from paths import FORECAST_ERROR_STATS
            self._err_stats = json.loads(FORECAST_ERROR_STATS.read_text())
        best, best_state, bad = -1.0, None, 0
        for epoch in range(1, cfg.epochs + 1):
            self.net.train()
            self._training = True
            order = np.random.permutation(train_idx)
            total = 0.0
            for s in range(0, len(order), cfg.batch):
                fr, ex, hs_t, lv = self._batch(order[s:s + cfg.batch])
                logits, hs_hat = self.net(fr, ex)
                mask = lv >= 0
                ce = F.cross_entropy(logits.permute(0, 1, 3, 4, 2)[mask], lv[mask], weight=weights)
                probs = torch.softmax(logits, 2)
                expected = (probs * levels).sum(2)
                ordinal = ((expected - lv.float()) ** 2)[mask].mean()
                loss = ce + 0.1 * ordinal
                if hs_hat is not None:
                    m2 = ~torch.isnan(hs_t)
                    loss = loss + 0.5 * F.huber_loss(hs_hat[m2] / 2, hs_t[m2] / 2, delta=1.0)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step(); sched.step()
                total += loss.item() * len(fr)
            self._training = False
            score = score_fn(self, val_idx)
            self.history.append({"epoch": epoch, "train_loss": total / len(order), "val_score": score})
            print(f"    epoch {epoch:02d}  loss {total / len(order):.4f}  val score {score:.4f}", flush=True)
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_val = best
        return self

    def predict_all(self, idx, batch=8, future_weather=None):
        """-> probs [N, LEADS, 5, H, W] (float16) and HotSpot forecast [N, LEADS, H, W] (or None).
        future_weather: optional {issue index: [8, H, W] scaled weather for the next two weeks},
        e.g. from archived weather forecasts instead of the observed weather."""
        import torch
        self._future = future_weather
        self.net.eval()
        H, W = self.builder.c.valid.shape
        probs, hss = [], []
        with torch.no_grad():
            for s in range(0, len(idx), batch):
                fr, ex = self._batch(idx[s:s + batch], with_targets=False)
                logits, hs_hat = self.net(fr, ex)
                probs.append(torch.softmax(logits, 2)[..., :H, :W].cpu().numpy().astype(np.float16))
                if hs_hat is not None:
                    hss.append(hs_hat[..., :H, :W].cpu().numpy().astype(np.float16))
        self._future = None
        return np.concatenate(probs), (np.concatenate(hss) if hss else None)

    def save(self, path):
        import torch
        torch.save({"state": self.net.state_dict(), "cfg": asdict(self.cfg)}, path)

    @classmethod
    def load(cls, path, cube, drivers):
        import torch
        ck = torch.load(path, map_location="cpu")
        obj = cls(Config(**ck["cfg"]))
        obj.dev = obj._device()
        obj.builder = Builder(cube, drivers, obj.cfg)
        obj.net = build_net(obj.builder.n_frame_channels(), obj.builder.n_extra_channels(), obj.cfg).to(obj.dev)
        obj.net.load_state_dict(ck["state"])
        return obj
