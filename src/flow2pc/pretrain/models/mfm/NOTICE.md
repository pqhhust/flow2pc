The MFM implementation under `_source/mfm/` is ported from the local
`metric-flow-matching` checkout at revision
`e44e03ec6207d2b111c1d51a03dcc53a98f87b95`, including its local
uncommitted changes at the time of the port. The upstream MIT license is
retained in `LICENSE`. Imports were rewritten to the Flow2PC package path;
the training algorithms and defaults were not intentionally changed.

The CITE-100 native entry point reads the original file from `flow2pc/data`
through a symlink in each run directory. It does not modify the data source.
