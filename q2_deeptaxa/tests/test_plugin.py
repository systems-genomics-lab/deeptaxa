"""Plugin registration tests.

These require a QIIME 2 environment and are skipped automatically when qiime2
is not importable.
"""

import unittest

try:
    import qiime2  # noqa: F401

    HAVE_QIIME2 = True
except ImportError:  # pragma: no cover - exercised only without qiime2
    HAVE_QIIME2 = False


@unittest.skipUnless(HAVE_QIIME2, "requires a QIIME 2 environment")
class TestPluginRegistration(unittest.TestCase):
    def setUp(self):
        from q2_deeptaxa.plugin_setup import plugin

        self.plugin = plugin

    def test_plugin_name(self):
        self.assertEqual(self.plugin.name, "deeptaxa")

    def test_actions_registered(self):
        self.assertIn("classify", self.plugin.methods)
        self.assertIn("fit", self.plugin.methods)
        self.assertIn("describe", self.plugin.visualizers)

    def test_semantic_type_registered(self):
        # plugin.types is keyed by the semantic type name.
        self.assertIn("DeepTaxaModel", self.plugin.types)

    def test_formats_registered(self):
        # plugin.formats is keyed by the format class name.
        registered = set(self.plugin.formats)
        self.assertIn("DeepTaxaModelFormat", registered)
        self.assertIn("DeepTaxaModelDirectoryFormat", registered)


if __name__ == "__main__":
    unittest.main()
