"""VRAM reserve for the knowledge layer (Human decision 2026-09-14).

The DPMtF knowledge layer's embedding server (LEANN/contriever) needs
1.6-2.2 GB of the card at search time. Every local GPU profile declares the
reserve and the card size, and every FreeToken alias budgets so that the
reserve stays free: its memory ratio leaves at least the reserve, and its
start gate tolerates the embedding server being resident.
"""
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RESERVE = 2200
CARD = 32607
LOCAL_GPU_PROFILES = ("local_ollama_cuda0", "freetoken_cuda0", "freetoken_qwen38_cuda0")


def _profiles():
    return yaml.safe_load((REPO_ROOT / "runtime_profiles.yaml").read_text())["runtime_profiles"]


def _models():
    return yaml.safe_load((REPO_ROOT / "models.yaml").read_text())["models"]


class VramReserveTests(unittest.TestCase):
    def test_local_gpu_profiles_declare_the_reserve_and_card(self):
        profiles = _profiles()
        for name in LOCAL_GPU_PROFILES:
            self.assertEqual(profiles[name].get("vram_reserve_mib"), RESERVE, name)
            self.assertEqual(profiles[name].get("gpu_total_mib"), CARD, name)

    def test_every_freetoken_alias_leaves_the_reserve_free(self):
        profiles = _profiles()
        seen = 0
        for alias, spec in _models().items():
            prof = profiles.get(spec.get("runtime_profile"), {})
            if prof.get("backend") != "freetoken" or "vram_reserve_mib" not in prof:
                continue
            seen += 1
            ratio = spec.get("memory_ratio")
            self.assertIsNotNone(ratio, f"{alias}: memory_ratio must be pinned")
            budget = (1.0 - float(ratio)) * prof["gpu_total_mib"]
            self.assertGreaterEqual(budget, prof["vram_reserve_mib"], alias)
            gate = spec.get("min_free_vram_mib")
            self.assertIsNotNone(gate, f"{alias}: min_free_vram_mib must be set")
            self.assertLessEqual(int(gate), prof["gpu_total_mib"] - prof["vram_reserve_mib"],
                                 f"{alias}: the start gate must tolerate a resident embedding server")
        self.assertGreaterEqual(seen, 4)

    def test_profile_schema_accepts_the_reserve_fields(self):
        from model_allocator import schema
        self.assertIs(schema.PROFILE_FIELDS["vram_reserve_mib"], int)
        self.assertIs(schema.PROFILE_FIELDS["gpu_total_mib"], int)


if __name__ == "__main__":
    unittest.main()
