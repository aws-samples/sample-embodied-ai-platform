# Protocols

This is a thin index of the data engine's pluggable layers. It names each protocol and points at its source; it does not re-derive method signatures here — read [`../contracts/protocols.py`](../contracts/protocols.py) for those.

Two rules that are non-obvious and will otherwise be got wrong:

1. **Import from `data.contracts`, never from a submodule.** Every container, protocol, and profile symbol is re-exported from the single import point `data.contracts` — [`../contracts/__init__.py`](../contracts/__init__.py)'s `__all__` is the literal enumeration of that surface.
2. **These are structural types, not base classes.** Every protocol below is a `typing.Protocol`. An implementation satisfies one purely by having matching method signatures — it never imports or inherits from `protocols.py`. Conformance is proved by a type checker against an annotated assignment (`impl: BaseInputAdapter = MyAdapter()`), not by `isinstance`. [`../tests/test_contracts.py`](../tests/test_contracts.py) is the live example of exactly that pattern — five stub implementations, each assigned to its protocol-typed variable with no inheritance in sight.

`BaseEnricher`/`BaseFilter` deliberately take an `EpisodeHeader`, not a `CanonicalEpisode` — see [`../contracts/containers.py`](../contracts/containers.py) for why: `CanonicalEpisode.frames` is a single-pass generator, and handing it to either protocol would invite double-iteration bugs.

## The five protocols, plus one decoder seam

| Protocol | Implement this when | Source | Containers exchanged |
|---|---|---|---|
| `BaseInputAdapter` | You're adding a new source format or a non-MCAP container. | [`../contracts/protocols.py`](../contracts/protocols.py) | Takes a path + `EmbodimentProfile`, returns a `CanonicalEpisode`. |
| `BaseSynchronizer` | You need a different clock/resample strategy than the shipped master-clock model. | [`../contracts/protocols.py`](../contracts/protocols.py) | Takes raw per-topic streams + `EmbodimentProfile`, returns a `CanonicalEpisode`. |
| `BaseEnricher` | You want a pluggable quality-scoring pass over episodes. | [`../contracts/protocols.py`](../contracts/protocols.py) | Takes an `EpisodeHeader`, returns an `EnrichmentPayload`. |
| `BaseFilter` | You want to keep/drop episodes by a pluggable rule. | [`../contracts/protocols.py`](../contracts/protocols.py) | Takes an `EpisodeHeader`, returns `bool`. |
| `BaseDatasetWriter` | You're adding a new output format alongside LeRobot v2.1. | [`../contracts/protocols.py`](../contracts/protocols.py) | Takes a `CanonicalEpisode`, returns an `EpisodeWriteResult`; also owns `finalize()`. |
| `JpegDecoderBackend` (additional row — a real pluggable seam, selected by name through [`../decoders.py`](../decoders.py) rather than by import) | You're deploying to an environment where the default JPEG decoder isn't the right choice (e.g. a Linux container). | [`../contracts/protocols.py`](../contracts/protocols.py) | Takes raw JPEG `bytes`, returns an HxWx3 uint8 RGB array. |

### Docstring excerpts

**`BaseInputAdapter`**
> Reads one source recording and produces one `CanonicalEpisode`.

**`BaseSynchronizer`**
> Builds the master-clock grid and yields a single-pass `CanonicalFrame` stream from a set of raw, per-topic streams.

**`BaseEnricher`**
> Produces a Layer-0 quality verdict for one episode. Takes an `EpisodeHeader`, deliberately not a `CanonicalEpisode`.

**`BaseFilter`**
> Decides whether to keep one episode. Takes an `EpisodeHeader`, deliberately not a `CanonicalEpisode`.

**`BaseDatasetWriter`**
> Streams a `CanonicalEpisode` to an on-disk dataset, one frame at a time.

**`JpegDecoderBackend`**
> Decodes compressed JPEG bytes to an HxWx3 uint8 RGB array. The pluggable seam behind `ImageRef.decode()`.

## Shipped implementations

| Protocol | Implementation | Source |
|---|---|---|
| `BaseInputAdapter` | `GalaxeaInputAdapter` | [`../adapters/galaxea.py`](../adapters/galaxea.py) |
| `BaseSynchronizer` | `MasterClockSynchronizer` | [`../sync/master_clock.py`](../sync/master_clock.py) |
| `BaseDatasetWriter` | `LeRobotV21Writer` | [`../writers/lerobot_v21.py`](../writers/lerobot_v21.py) — reachable by format key through its `get_writer` factory. |
| `JpegDecoderBackend` | `SimplejpegBackend` (the `simplejpeg`-based default) | [`../decoders.py`](../decoders.py) |

**No `BaseEnricher` or `BaseFilter` implementation ships today.** They are declared, reserved seams for a future quality-scoring or filtering pass — their absence is by design, not a gap to fill before this pipeline is usable.
