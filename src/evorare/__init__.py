"""evorare — realized-sample diversity-trend diagnostic for LLM evolutionary-search archives.

evorare describes the realized-sample diversity trend of an archive using ecology
Hill numbers and Rao quadratic entropy, and routes which estimators are statistically
valid for the archive before they drive an early-stopping decision. It does NOT estimate
population diversity, and it is a diagnostic instrument, not a stopping guarantee.
"""

__version__ = "0.1.0a2"

__all__ = ["__version__"]
