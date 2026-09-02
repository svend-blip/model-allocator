"""`reasoning_effort` is a known alias field (2026-09-02): it survives
schema validation and reaches the resolved alias for clients that send it."""
import unittest

from model_allocator.schema import COMMON_ALIAS_FIELDS


class ReasoningEffortField(unittest.TestCase):
    def test_field_is_declared_as_a_string(self):
        self.assertIs(COMMON_ALIAS_FIELDS.get("reasoning_effort"), str)


if __name__ == "__main__":
    unittest.main()
