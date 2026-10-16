from realestate import db as db_module
from realestate import scoring
from realestate import sql_builder
from realestate.formatting import format_response
from realestate.models import PropertyResult
from realestate.query_analysis import QueryAnalyzer


class RealEstateSearchService:

    def __init__(self, db_path: str = "gradovi.db", use_llm: bool = True, llm_model: str = "gpt-4o-mini"):
        self.db_path = db_path
        self.use_llm = use_llm
        self.analyzer = QueryAnalyzer(db_path, use_llm=use_llm, llm_model=llm_model)
        self.column_stats = self.analyzer.column_stats

    def search(self, user_input: str, top_n: int = 5, detailed: bool = True) -> str:
        params = self.analyzer.analyze(user_input)
        try:
            sql, order_direction, conditions = sql_builder.build_sql_from_params(params, self.db_path)
        except ValueError as e:
            print(f" Query builder error: {e} — falling back to a default price-sorted query.")
            order_direction = "DESC"
            sql = sql_builder.build_select_query("1=1", order_direction)
            conditions = []
        params.weights = self.analyzer.auto_adjust_weights(params)
        weights = scoring.params_to_weights(params)

        user_specified_states = bool(params.states)
        results = db_module.run_query(self.db_path, sql)
        fallback_msg = ""

        if not results:
            results, fallback_msg = sql_builder.apply_fallback_logic(
                conditions, params, order_direction, user_specified_states, self.db_path
            )

            if not results:
                return (
                    f" No locations found in {', '.join(params.states)}. Please check the state name."
                    if user_specified_states else "  No results found."
                )

        properties = []
        for r in results:
            prop = PropertyResult(*r)
            prop.score = scoring.calculate_score(self.column_stats, prop, weights, params)
            properties.append(prop)

        top = scoring.sort_results_by_mode(properties, params, top_n)

        return format_response(top, fallback_msg, user_specified_states, params, len(results), top_n)
