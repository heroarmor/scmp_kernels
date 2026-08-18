"""k_bands table parsing and validation.

The whole point of the section is that a mispriced allocation is invisible
downstream -- the runtime prices the assignment rather than measuring the
kernel, so a cell that overspends its budget simply looks like a win. Every
invariant that keeps that from happening is enforced at load, and tested here.

CPU only: this exercises the config layer, not the kernels.
"""
import json
import os
import tempfile
import unittest

from scmp_kernels.mp import AdaptiveMPConfig

LEVELS = [128, 96, 64, 32]
WIDTH = 1152                                 # 9 chunks of 128
CHUNK_D = 128
CHUNK_BANDS = [0, 0, 1, 1, 2, 2, 3, 3, 3]    # widths 256/256/256/384
# Solved against those widths: (2/9)(a0+a1+a2) + (1/3)a3 == parent per rung.
WIDE = [144, 112, 80, 40]
NARROW = [96, 64, 32, 16]
OP = "down_proj"


def load(*, ladders=None, chunk_bands=None, n_bands=4, residual_width=None,
         chunk_d=CHUNK_D, levels=None):
    levels = LEVELS if levels is None else levels
    payload = {
        "stoc_len_levels": levels,
        "timestep_buckets": 1,
        "layer_buckets": 1,
        "buckets": {
            f"{OP}:t0:l0": {"thresholds": [0.75, 0.5, 0.25][: len(levels) - 1]},
        },
        "k_bands": {
            "n_bands": n_bands,
            "chunk_d": chunk_d,
            "residual_width": ({OP: WIDTH} if residual_width is None
                               else residual_width),
            "chunk_bands": ({f"{OP}:0": CHUNK_BANDS} if chunk_bands is None
                            else chunk_bands),
            "ladders": ({f"{OP}:t0:l0": [WIDE, WIDE, WIDE, NARROW]}
                        if ladders is None else ladders),
        },
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    try:
        return AdaptiveMPConfig(stoc_len_levels=list(levels),
                                threshold_table_path=path)
    finally:
        os.unlink(path)


class TestKBands(unittest.TestCase):

    def test_loads_and_exposes_bands(self):
        cfg = load()
        self.assertEqual(cfg.k_band_count, 4)
        self.assertEqual(cfg.k_band_chunk_d, CHUNK_D)
        bands, ladders = cfg.get_k_bands(OP, 0, 28)
        self.assertEqual(bands, CHUNK_BANDS)
        self.assertEqual(ladders, [WIDE, WIDE, WIDE, NARROW])

    def test_absent_section_leaves_path_disabled(self):
        cfg = AdaptiveMPConfig(stoc_len_levels=list(LEVELS))
        self.assertEqual(cfg.k_band_count, 0)
        self.assertIsNone(cfg.get_k_bands(OP, 0, 28))

    def test_uncovered_operator_runs_the_parent(self):
        self.assertIsNone(load().get_k_bands("q_proj", 0, 28))

    def test_parent_ladder_in_every_band_is_iso_compute(self):
        """L[b][k] == parent[k] must satisfy the identity exactly.

        This is the property that makes the refinement unable to lose: the
        per-row parent has to be a legal point in the space.
        """
        parent = [list(LEVELS) for _ in range(4)]
        _, ladders = load(ladders={f"{OP}:t0:l0": parent}).get_k_bands(OP, 0, 28)
        self.assertEqual(ladders, parent)

    def test_overspend_is_rejected(self):
        over = [[160, 112, 80, 40], WIDE, WIDE, NARROW]
        with self.assertRaisesRegex(ValueError, "overspends rung 0"):
            load(ladders={f"{OP}:t0:l0": over})

    def test_large_underspend_is_rejected(self):
        half = [[x // 2 for x in WIDE] for _ in range(3)]
        half.append([x // 2 for x in NARROW])
        with self.assertRaisesRegex(ValueError, "underspends rung 0"):
            load(ladders={f"{OP}:t0:l0": half})

    def test_band_with_one_chunk_is_rejected(self):
        """Every band needs >= 2 chunks to stay on the chunked kernel path."""
        lonely = [0, 0, 1, 1, 1, 2, 2, 2, 3]
        with self.assertRaisesRegex(ValueError, "band 3 with 1 chunk"):
            load(chunk_bands={f"{OP}:0": lonely})

    def test_band_count_is_per_operator(self):
        """n_bands is a MAXIMUM; a narrow operator may use fewer.

        Forcing one global count silently drops every narrow projection out of
        the banded path, which reads as "bands did not help" rather than as a
        configuration error.
        """
        cfg = load(n_bands=8)
        _, ladders = cfg.get_k_bands(OP, 0, 28)
        self.assertEqual(len(ladders), 4)

    def test_skipped_band_id_is_rejected(self):
        skipped = [0, 0, 1, 1, 3, 3, 3, 3, 3]
        with self.assertRaisesRegex(ValueError, "skip a band"):
            load(chunk_bands={f"{OP}:0": skipped})

    def test_wrong_chunk_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "chunk entries"):
            load(chunk_bands={f"{OP}:0": CHUNK_BANDS[:-1]})

    def test_ladder_rung_count_must_match_levels(self):
        short = [WIDE[:-1], WIDE[:-1], WIDE[:-1], NARROW[:-1]]
        with self.assertRaisesRegex(ValueError, "rungs, expected 4"):
            load(ladders={f"{OP}:t0:l0": short})

    def test_chunk_bands_without_ladders_is_rejected(self):
        """Silently falling back to the parent is the failure to prevent."""
        with self.assertRaisesRegex(ValueError, "no.*ladders entry"):
            load(ladders={})

    def test_missing_residual_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "residual_width is required"):
            load(residual_width={})

    def test_band_widths_must_agree_across_blocks(self):
        """Ladders are shared across an operator's blocks, so widths must be."""
        other = [0, 0, 0, 1, 1, 2, 2, 3, 3]
        with self.assertRaisesRegex(ValueError, "band widths"):
            load(chunk_bands={f"{OP}:0": CHUNK_BANDS, f"{OP}:1": other})

    def test_n_bands_below_two_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "n_bands must be >= 2"):
            load(n_bands=1)

    def test_tail_chunk_is_priced_at_its_real_width(self):
        """A short trailing chunk must not be charged a full chunk_d."""
        # 1160 channels -> nine 128-wide chunks plus an 8-wide tail.
        bands = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3]
        parent = [list(LEVELS) for _ in range(4)]
        cfg = load(residual_width={OP: 1160},
                   chunk_bands={f"{OP}:0": bands},
                   ladders={f"{OP}:t0:l0": parent})
        self.assertIsNotNone(cfg.get_k_bands(OP, 0, 28))
        # The same map priced against 1152 has the wrong chunk count.
        with self.assertRaisesRegex(ValueError, "chunk entries"):
            load(residual_width={OP: WIDTH}, chunk_bands={f"{OP}:0": bands},
                 ladders={f"{OP}:t0:l0": parent})


if __name__ == "__main__":
    unittest.main(verbosity=2)
