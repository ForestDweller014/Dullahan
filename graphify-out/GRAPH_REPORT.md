# Graph Report - .  (2026-07-15)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1014 nodes · 2848 edges · 73 communities (48 shown, 25 thin omitted)
- Extraction: 71% EXTRACTED · 29% INFERRED · 0% AMBIGUOUS · INFERRED: 832 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0580a6c7`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 42
- Community 43
- Community 44
- Community 46
- Community 47
- Community 48
- Community 49
- Community 51
- Community 52
- Community 53
- Community 54
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 70
- Community 71
- Community 72

## God Nodes (most connected - your core abstractions)
1. `ContextBundle` - 64 edges
2. `QueryEnvelope` - 60 edges
3. `AgentRuntime` - 48 edges
4. `ExpertResponse` - 42 edges
5. `ContextDocument` - 40 edges
6. `InferenceConfig` - 38 edges
7. `KeywordEmbeddingModel` - 38 edges
8. `DeviceInventory` - 34 edges
9. `AgentRuntimeConfig` - 32 edges
10. `StubSynthesisProvider` - 30 edges

## Surprising Connections (you probably didn't know these)
- `AgentRuntime` --uses--> `EmbeddingModel`  [INFERRED]
  apps/agent-runtime/src/agent_runtime/agent.py → packages/shared/src/dullahan_shared/embeddings.py
- `AgentRuntime` --uses--> `ContextBundle`  [INFERRED]
  apps/agent-runtime/src/agent_runtime/agent.py → packages/shared/src/dullahan_shared/schemas/context.py
- `AgentRuntime` --uses--> `ExpertResponse`  [INFERRED]
  apps/agent-runtime/src/agent_runtime/agent.py → packages/shared/src/dullahan_shared/schemas/expert.py
- `AgentRuntime` --uses--> `QueryEnvelope`  [INFERRED]
  apps/agent-runtime/src/agent_runtime/agent.py → packages/shared/src/dullahan_shared/schemas/query.py
- `AgentRuntime` --uses--> `TokenCounter`  [INFERRED]
  apps/agent-runtime/src/agent_runtime/agent.py → packages/shared/src/dullahan_shared/tokenization.py

## Import Cycles
- None detected.

## Communities (73 total, 25 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (72): AgentRuntime, OpenAICompatibleSynthesisProvider, RuntimeError, Final-answer provider for OpenAI-compatible completion APIs., ResponseAggregator, SynthesisProvider, SynthesisProviderError, SynthesisRequest (+64 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (69): generate_clusters(), Path, _ensure_role_context_document(), _expert_for_cluster(), generate_experts_from_clusters(), _infer_repo_root(), Any, Path (+61 more)

### Community 2 - "Community 2"
Cohesion: 0.12
Nodes (57): activate_model(), ActivateRequest, Backend, checked_name(), deactivate(), default_supported_backends(), delete_adapter(), delete_model() (+49 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (34): build_parser(), _hook_path(), install(), InstallError, main(), patch_hook_text(), ArgumentParser, Path (+26 more)

### Community 4 - "Community 4"
Cohesion: 0.14
Nodes (9): T, ThreadSafeList, RecursionGuard, InMemoryTraceCollector, test_recursion_guard_rejects_duplicate_query_signature(), new_id(), Create a stable, readable identifier for runtime envelopes., ExecutionSpan (+1 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (20): BenchmarkCase, BenchmarkRun, main(), _json_request(), _mac_system_used_bytes(), memory_snapshot(), MemorySampler, MemorySnapshot (+12 more)

### Community 6 - "Community 6"
Cohesion: 0.20
Nodes (26): InferenceConfig, DeviceInventory, resolve_inference_plan(), _resolve_quantization(), inventory(), test_cpu_plan_fails_early_when_quantized_model_exceeds_ram(), test_cuda_plan_fails_when_host_cannot_absorb_vram_shortfall(), test_cuda_shortfall_uses_offload_instead_of_more_quantization() (+18 more)

### Community 7 - "Community 7"
Cohesion: 0.16
Nodes (10): HttpCalTool, FakeHttpResponse, test_http_cal_tool_posts_augment_request(), test_http_cal_tool_posts_batch_augment_request(), test_http_edl_tool_posts_batch_dispatch_request(), test_http_edl_tool_posts_dispatch_request(), test_http_tool_raises_for_http_error(), BaseModel (+2 more)

### Community 8 - "Community 8"
Cohesion: 0.17
Nodes (16): CompletionRequest, create_ollama_app(), EmbeddingRequest, BaseModel, FastAPI, TokenizeRequest, OllamaClient, OllamaEmbeddingResult (+8 more)

### Community 9 - "Community 9"
Cohesion: 0.24
Nodes (15): LocalCalTool, augment_context(), augment_context_batch(), AugmentContextRequest, AugmentContextResponse, BatchAugmentContextRequest, BatchAugmentContextResponse, BaseModel (+7 more)

### Community 10 - "Community 10"
Cohesion: 0.15
Nodes (10): EmbeddingModel, Protocol, LocalWorldStateDB, Path, Local long-term memory store used by CAL during context augmentation., LocalVectorIndex, BaseModel, Path (+2 more)

### Community 11 - "Community 11"
Cohesion: 0.27
Nodes (11): ExpertAttentionScore, ExpertRoute, BaseModel, ExpertRunner, ExpertPromptBuilder, StubModelProvider, test_expert_runner_builds_prompt_and_records_model_metadata(), test_expert_runner_overrides_local_expert_alias_for_hosted_model() (+3 more)

### Community 12 - "Community 12"
Cohesion: 0.14
Nodes (5): Hierarchical agent runtime., test_cal_openai_environment_sets_hosted_models(), cpu_inference_base_url(), _free_port(), Path

### Community 13 - "Community 13"
Cohesion: 0.17
Nodes (9): PostgresWorldStateConfig, BaseModel, FakeConnection, FakeCursor, FakeDocumentSource, test_postgres_world_state_rebuild_writes_graph_documents_to_pgvector(), test_postgres_world_state_rejects_stale_embedding_dimensions(), test_postgres_world_state_rejects_unsafe_table_names() (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.26
Nodes (4): EdlConfig, BaseModel, ExpertDispatchService, test_edl_openai_environment_replaces_local_expert_model_alias()

### Community 15 - "Community 15"
Cohesion: 0.14
Nodes (14): DevicePreference, EmbeddingConfig, InferenceServerConfig, ModelCatalogConfig, ModelExportMode, ModelServerConfig, ModelServerEndpointConfig, OffloadConfig (+6 more)

### Community 16 - "Community 16"
Cohesion: 0.22
Nodes (14): dispatch(), dispatch_batch(), BatchDispatchRequest, BatchDispatchResponse, DispatchRequest, DispatchResponse, BaseModel, build_service() (+6 more)

### Community 17 - "Community 17"
Cohesion: 0.16
Nodes (15): build_parser(), main(), ArgumentParser, run_from_args(), Path, activate_model_server(), _admin_request(), export_model_server_package() (+7 more)

### Community 18 - "Community 18"
Cohesion: 0.22
Nodes (10): build_cal_handler(), build_cal_stdio_server(), build_edl_handler(), build_edl_stdio_server(), cal_stdio_main(), edl_stdio_main(), JsonRpcMcpServer, Any (+2 more)

### Community 19 - "Community 19"
Cohesion: 0.15
Nodes (11): test_real_embedding_model_produces_semantic_similarity(), cosine_similarity(), EmbeddingError, OpenAICompatibleEmbeddingModel, RuntimeError, Semantic embedding client for the Dullahan inference boundary., FakeResponse, test_cosine_similarity_normalizes_vectors_and_rejects_mismatched_dimensions() (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.20
Nodes (11): CalMcpHandler, EdlMcpHandler, _query_like(), FakeCalTool, FakeEdlTool, test_send_to_cal_handler_validates_and_returns_json_shape(), test_send_to_edl_handler_validates_and_returns_json_shape(), test_stdio_server_calls_cal_tool_and_returns_structured_content() (+3 more)

### Community 21 - "Community 21"
Cohesion: 0.27
Nodes (14): adapter_package(), add_adapter(), archive_bytes(), Path, test_activation_command_loads_named_base_and_lora_adapters(), test_adapter_crud_uploads_and_deletes_named_lora_weights(), test_archive_traversal_is_rejected(), test_cuda_quantization_defaults_to_cuda_only() (+6 more)

### Community 22 - "Community 22"
Cohesion: 0.26
Nodes (8): ExecutionArtifactStore, _mermaid_id(), _mermaid_label(), Path, _safe_path_id(), _short_label(), AgentRunResult, BaseModel

### Community 23 - "Community 23"
Cohesion: 0.19
Nodes (11): estimate_tokens(), pack_documents_to_budget(), _truncate_to_token_budget(), merge_ranked_documents(), test_pack_documents_to_budget_keeps_ranked_documents_that_fit(), test_pack_documents_to_budget_returns_empty_for_non_positive_budget(), test_pack_documents_to_budget_truncates_first_document_if_needed(), ContextDocument (+3 more)

### Community 24 - "Community 24"
Cohesion: 0.17
Nodes (9): test_real_token_counter_matches_ollama_prompt_usage(), InferenceTokenCounter, RuntimeError, Counts tokens through the serving model's native tokenizer usage., TokenizationError, FakeResponse, test_inference_token_counter_rejects_missing_native_count(), test_inference_token_counter_supports_gateway_bearer_auth() (+1 more)

### Community 25 - "Community 25"
Cohesion: 0.15
Nodes (7): AttentionRouter, ExpertRegistry, Any, Path, _ExpertConcurrencyLimiter, test_attention_router_returns_softmax_distribution(), test_expert_registry_loads_role_contexts()

### Community 26 - "Community 26"
Cohesion: 0.34
Nodes (10): Path, InferenceProvider, embedding_model(), generation_model(), inference_api_key(), inference_base_url(), inference_provider(), tokenizer_api_key() (+2 more)

### Community 27 - "Community 27"
Cohesion: 0.31
Nodes (11): build_parser(), format_text(), main(), ArgumentParser, run_from_args(), install_fake_local_runtime(), test_cli_json_output(), test_cli_remote_transport_uses_http_runtime() (+3 more)

### Community 28 - "Community 28"
Cohesion: 0.19
Nodes (4): FakeConnection, FakeCursor, FakePsycopg, test_export_postgres_context_writes_markdown_collection()

### Community 29 - "Community 29"
Cohesion: 0.28
Nodes (7): _index_name(), _metadata_dict(), PostgresWorldStateDB, PostgreSQL + pgvector-backed long-term memory store for CAL retrieval., _to_pgvector(), _validate_table_name(), test_pgvector_literal_formats_embedding_for_sql_cast()

### Community 30 - "Community 30"
Cohesion: 0.24
Nodes (6): ModelTokenizer, Lazy, thread-safe access to the generation model's exact tokenizer., FakeEncoding, FakeTokenizer, test_model_tokenizer_counts_encoded_ids_and_reuses_instance(), Tokenizer

### Community 31 - "Community 31"
Cohesion: 0.36
Nodes (9): _apple_metal_inventory(), detect_device(), _nvidia_smi_inventory(), _system_memory_gb(), _torch_cuda_inventory(), test_auto_device_falls_back_to_cpu(), test_auto_device_uses_apple_metal_when_available(), test_device_inventory_normalizes_string_values() (+1 more)

### Community 32 - "Community 32"
Cohesion: 0.25
Nodes (6): _safe_model_id(), Local WorldStateDB implementation., Path, test_world_state_builds_and_reuses_persistent_vector_index(), test_world_state_search_respects_top_k(), test_world_state_search_returns_ranked_memory_documents()

### Community 33 - "Community 33"
Cohesion: 0.67
Nodes (9): QuantizationMode, _estimated_model_memory_gb(), _memory_plan(), _model_for_quantization(), _model_server_plan(), _ollama_plan(), _quantization_bits(), _usable_system_memory_gb() (+1 more)

### Community 34 - "Community 34"
Cohesion: 0.32
Nodes (5): Path, test_agent_runtime_loads_recursion_config(), test_agent_runtime_config_reads_planner_environment(), test_agent_runtime_config_uses_shared_openai_environment(), test_openai_configuration_requires_bearer_token()

### Community 35 - "Community 35"
Cohesion: 0.39
Nodes (4): Counter, LexicalRetriever, T, RankedItem

### Community 36 - "Community 36"
Cohesion: 0.32
Nodes (4): ContextSource, StrEnum, GraphDocumentSource, Path

### Community 37 - "Community 37"
Cohesion: 0.38
Nodes (4): CalConfig, BaseModel, Path, test_cal_openai_mode_keeps_dullahan_tokenize_boundary()

### Community 71 - "Community 71"
Cohesion: 0.14
Nodes (9): Compatibility wrapper returning only the synthesized answer text., HttpEdlTool, HttpToolError, post_json(), BaseModel, RuntimeError, LocalEdlTool, ExpertResponse (+1 more)

### Community 72 - "Community 72"
Cohesion: 0.20
Nodes (7): OpenAICompatibleHttpProvider, HTTP provider for Dullahan completions or hosted OpenAI Responses., FakeHttpResponse, test_edl_config_rejects_unknown_model_provider(), test_edl_config_selects_http_model_provider(), test_openai_compatible_http_provider_posts_completion_request(), test_openai_compatible_http_provider_requires_native_token_usage()

## Knowledge Gaps
- **3 isolated node(s):** `build-base.sh script`, `entrypoint.sh script`, `dullahan`
  These have ≤1 connection - possible missing edges or undocumented components.
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `create_ollama_app()` connect `Community 8` to `Community 17`, `Community 2`, `Community 6`, `Community 30`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `lifespan()` connect `Community 2` to `Community 8`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Why does `EmbeddingModel` connect `Community 10` to `Community 0`, `Community 1`, `Community 32`, `Community 37`, `Community 9`, `Community 11`, `Community 12`, `Community 13`, `Community 14`, `Community 25`, `Community 29`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Are the 51 inferred relationships involving `ContextBundle` (e.g. with `AgentRuntime` and `AgentRunRequest`) actually correct?**
  _`ContextBundle` has 51 INFERRED edges - model-reasoned connections that need verification._
- **Are the 42 inferred relationships involving `QueryEnvelope` (e.g. with `AgentRuntime` and `OpenAICompatibleSynthesisProvider`) actually correct?**
  _`QueryEnvelope` has 42 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `AgentRuntime` (e.g. with `OpenAICompatibleSynthesisProvider` and `ResponseAggregator`) actually correct?**
  _`AgentRuntime` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `ExpertResponse` (e.g. with `AgentRuntime` and `OpenAICompatibleSynthesisProvider`) actually correct?**
  _`ExpertResponse` has 29 INFERRED edges - model-reasoned connections that need verification._