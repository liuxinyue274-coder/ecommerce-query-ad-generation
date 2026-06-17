# V5 LLM Dynamic Creative Schemas

## query_normalization
- input
  - `raw_query: str`
- output
  - `raw_query: str`
  - `normalized_query: str`
  - `query_length: int`
  - `alpha_num_tokens: list[str]`
  - `han_tokens: list[str]`
  - `flags: dict[str, bool]`

## intent_enricher
- input
  - `query_bundle: query_normalization.output`
  - `retrieval_result: infer.retrieve_evidence_bundle.result`
- output
  - `category_hint: str`
  - `brand_hint: str`
  - `model_hint: str`
  - `scene_hint: str`
  - `crowd_hint: str`
  - `price_intent: str`
  - `attribute_hints: list[str]`
  - `search_focus: list[str]`
  - `intent_summary: str`

## evidence_selector
- input
  - `intent_bundle: intent_enricher.output`
  - `retrieval_result: infer.retrieve_evidence_bundle.result`
  - `top_k: int`
- output
  - `retrieval_source: str`
  - `anchor_item: dict[str, object]`
  - `selected_evidence_items: list[dict[str, object]]`
  - `fact_block: list[str]`
  - `selection_reason: str`

## user_profile_builder
- input
  - `user_id: Optional[int]`
  - `retrieval_result: infer.retrieve_evidence_bundle.result`
  - `users_map: Optional[dict[int, dict[str, object]]]`
- output
  - `user_id: Optional[int]`
  - `profile_available: bool`
  - `persona_summary: str`
  - `demographic_tags: list[str]`
  - `behavior_tags: list[str]`
  - `recent_behavior_titles: list[str]`
  - `personalization_strength: str`
  - `user_profile_raw: dict[str, object]`

## style_retriever
- input
  - `query_bundle: query_normalization.output`
  - `intent_bundle: intent_enricher.output`
  - `user_profile_bundle: user_profile_builder.output`
  - `requested_tone: str`
- output
  - `style_id: str`
  - `tone: str`
  - `length_range: str`
  - `style_rules: list[str]`
  - `negative_rules: list[str]`
  - `example_pattern: str`

## prompt_builder
- input
  - `query_bundle: query_normalization.output`
  - `intent_bundle: intent_enricher.output`
  - `evidence_bundle: evidence_selector.output`
  - `user_profile_bundle: user_profile_builder.output`
  - `style_bundle: style_retriever.output`
  - `candidate_count: int`
- output
  - `system_prompt: str`
  - `user_prompt: str`
  - `full_prompt: str`
  - `output_format: str`
  - `candidate_count: int`
  - `prompt_sections: dict[str, object]`

## llm_provider
- input
  - `prompt_bundle: prompt_builder.output`
  - `llm_config_path: Optional[path]`
  - `candidate_count: int`
- output
  - `provider: str`
  - `model_name: str`
  - `status: str`
  - `requested_candidates: int`
  - `raw_generations: list[dict[str, str]]`
  - `fallback_used: bool`

## llm_output_parser
- input
  - `provider_bundle: llm_provider.output`
- output
  - `parse_status: str`
  - `parsed_candidates: list[dict[str, object]]`

## copy_validator
- input
  - `parsed_bundle: llm_output_parser.output`
  - `query_bundle: query_normalization.output`
  - `intent_bundle: intent_enricher.output`
  - `evidence_bundle: evidence_selector.output`
  - `style_bundle: style_retriever.output`
- output
  - `validated_candidates: list[dict[str, object]]`
  - `validator_summary: dict[str, object]`

## copy_ranker
- input
  - `validated_bundle: copy_validator.output`
  - `intent_bundle: intent_enricher.output`
  - `evidence_bundle: evidence_selector.output`
- output
  - `ranked_candidates: list[dict[str, object]]`
  - `top_candidate: dict[str, object]`

## copy_rewriter
- input
  - `ranked_bundle: copy_ranker.output`
  - `query_bundle: query_normalization.output`
  - `intent_bundle: intent_enricher.output`
  - `evidence_bundle: evidence_selector.output`
  - `user_profile_bundle: user_profile_builder.output`
- output
  - `rewritten: bool`
  - `rewrite_reason: str`
  - `final_candidate: dict[str, object]`

## final_ad_copy
- input
  - `rewriter_bundle: copy_rewriter.output`
- output
  - `final_ad_copy: str`
  - `final_source: str`
  - `final_candidate: dict[str, object]`
