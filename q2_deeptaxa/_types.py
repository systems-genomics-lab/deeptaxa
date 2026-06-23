"""
Semantic types for the q2-deeptaxa plugin.

``DeepTaxaModel`` is a standalone, trained hierarchical taxonomy classifier. It
is registered to :class:`DeepTaxaModelDirectoryFormat`, so an artifact of this
type is a directory containing a single ``model.pt`` checkpoint.
"""

from qiime2.plugin import SemanticType

DeepTaxaModel = SemanticType("DeepTaxaModel")
