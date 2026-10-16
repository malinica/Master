from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

from realestate import stats as stats_module
from realestate.models import QueryParameters
from realestate.prompts import build_query_analysis_prompt


class QueryAnalyzer:
    def __init__(self, db_path: str, use_llm: bool = True, llm_model: str = "gpt-4o-mini"):
        self.db_path = db_path
        self.use_llm = use_llm
        self.price_stats = stats_module.get_price_statistics(db_path)
        self.column_stats = stats_module.get_all_column_statistics(db_path)

        if self.use_llm:
            self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
            self.parser = PydanticOutputParser(pydantic_object=QueryParameters)
            self.prompt_template = build_query_analysis_prompt(
                self.parser, self.price_stats, self.column_stats
            )

    def analyze(self, user_input: str) -> QueryParameters:
        if not self.use_llm:
            return self.fallback_analysis(user_input)

        try:
            chain = self.prompt_template | self.llm | self.parser
            result = chain.invoke({"query": user_input})

            user_lower = user_input.lower()

            if "average" in user_lower and "price" in user_lower:
                result.extreme_mode = "average"
                result.weights = {"price": 1.0, "hospital": 0.0, "temperature": 0.0, "weather": 0.0, "schools": 0.0}
                result.price_min = None
                result.price_max = None

            if any(word in user_lower for word in ["hottest", "warmest"]):
                result.extreme_mode = "max"
                result.weights = {"price": 0.0, "hospital": 0.0, "temperature": 1.0, "weather": 0.0, "schools": 0.0}
                result.temperature_min = None
                result.temperature_max = None

            if any(word in user_lower for word in ["coldest", "coolest"]):
                result.extreme_mode = "min"
                result.weights = {"price": 0.0, "hospital": 0.0, "temperature": 1.0, "weather": 0.0, "schools": 0.0}
                result.temperature_min = None
                result.temperature_max = None

            if any(word in user_lower for word in ["most expensive", "priciest", "expensive"]) and "most" in user_lower or "priciest" in user_lower:
                result.extreme_mode = "max"
                result.weights = {"price": 1.0, "hospital": 0.0, "temperature": 0.0, "weather": 0.0, "schools": 0.0}
                result.price_min = None
                result.price_max = None

            if any(word in user_lower for word in ["cheapest", "most affordable", "lowest price"]):
                result.extreme_mode = "min"
                result.weights = {"price": 1.0, "hospital": 0.0, "temperature": 0.0, "weather": 0.0, "schools": 0.0}
                result.price_min = None
                result.price_max = None

            if any(word in user_lower for word in ["snowy", "snowiest"]):
                result.extreme_mode = "max"
                result.weather_preferences = ["snowy"]
                result.weights = {"price": 0.0, "hospital": 0.0, "temperature": 0.0, "weather": 1.0, "schools": 0.0}

            if any(word in user_lower for word in ["rainy", "rainiest"]):
                result.extreme_mode = "max"
                result.weather_preferences = ["rainy"]
                result.weights = {"price": 0.0, "hospital": 0.0, "temperature": 0.0, "weather": 1.0, "schools": 0.0}

            if any(word in user_lower for word in ["windy", "windiest", "wind", "strong wind"]) and "speed" not in user_lower:
                result.extreme_mode = "max"
                result.wind_speed_min = None
                result.weights = {"price": 0.0, "hospital": 0.0, "temperature": 0.0, "weather": 1.0, "schools": 0.0}

            if any(word in user_lower for word in ["humid", "humidity", "most humid"]):
                result.extreme_mode = "max"
                result.humidity_min = None
                result.weights = {"price": 0.0, "hospital": 0.0, "temperature": 0.0, "weather": 1.0, "schools": 0.0}

            if "school" in user_lower or "schools" in user_lower:
                if "100km" in user_lower or "100 km" in user_lower or "far" in user_lower:
                    result.schools_distance = "far"
                    result.schools_required = True
                elif "nearby" in user_lower or "10km" in user_lower or "10 km" in user_lower or "close" in user_lower:
                    result.schools_distance = "nearby"
                    result.schools_required = True
                elif "20km" in user_lower or "20 km" in user_lower or "medium" in user_lower:
                    result.schools_distance = "medium"
                    result.schools_required = True

                if "high" in user_lower:
                    result.school_types = ["high"]
                elif "elementary" in user_lower:
                    result.school_types = ["elementary"]
                elif "prekindergarten" in user_lower or "preschool" in user_lower:
                    result.school_types = ["prekindergarten"]

                if "most" in user_lower:
                    result.weights = {"price": 0.0, "hospital": 0.0, "temperature": 0.0, "weather": 0.0, "schools": 1.0}

            if result.extreme_mode:
                if any(word in user_lower for word in ["hottest", "coldest", "warmest", "coolest"]):
                    result.weights = {"price": 0.0, "hospital": 0.0, "temperature": 1.0, "weather": 0.0, "schools": 0.0}
                elif any(word in user_lower for word in ["expensive", "cheapest", "priciest", "price"]):
                    result.weights = {"price": 1.0, "hospital": 0.0, "temperature": 0.0, "weather": 0.0, "schools": 0.0}

            if not result.weights or len(result.weights) == 0:
                result.weights = {
                    'price': 0.2, 'hospital': 0.2, 'temperature': 0.2,
                    'weather': 0.2, 'schools': 0.2
                }

            for key in ['price', 'hospital', 'temperature', 'weather', 'schools']:
                if key not in result.weights:
                    result.weights[key] = 0.2

            return result

        except Exception as e:
            print(f"LLM Error: {e}")
            return self.fallback_analysis(user_input)

    def fallback_analysis(self, user_input: str) -> QueryParameters:
        params = QueryParameters()
        user_lower = user_input.lower()

        state_names = [
            "California", "Texas", "Florida", "New York", "Pennsylvania", "Illinois", "Ohio",
            "Georgia", "North Carolina", "Michigan", "New Jersey", "Virginia", "Washington",
            "Arizona", "Massachusetts", "Tennessee", "Indiana", "Missouri", "Maryland",
            "Wisconsin", "Colorado", "Minnesota", "South Carolina", "Alabama", "Louisiana",
            "Kentucky", "Oregon", "Oklahoma", "Connecticut", "Utah", "Iowa", "Nevada",
            "Arkansas", "Mississippi", "Kansas", "New Mexico", "Nebraska", "West Virginia",
            "Idaho", "Hawaii", "New Hampshire", "Maine", "Montana", "Rhode Island", "Delaware",
            "South Dakota", "North Dakota", "Alaska", "Vermont", "Wyoming"
        ]

        state_abbrev = {
            "ca": "California", "tx": "Texas", "fl": "Florida", "ny": "New York",
            "pa": "Pennsylvania", "il": "Illinois", "oh": "Ohio", "ga": "Georgia",
            "nc": "North Carolina", "mi": "Michigan", "nj": "New Jersey", "va": "Virginia",
            "wa": "Washington", "az": "Arizona", "ma": "Massachusetts", "tn": "Tennessee",
            "in": "Indiana", "mo": "Missouri", "md": "Maryland", "wi": "Wisconsin",
            "co": "Colorado", "mn": "Minnesota", "sc": "South Carolina", "al": "Alabama",
            "la": "Louisiana", "ky": "Kentucky", "or": "Oregon", "ok": "Oklahoma",
            "ct": "Connecticut", "ut": "Utah", "ia": "Iowa", "nv": "Nevada",
            "ar": "Arkansas", "ms": "Mississippi", "ks": "Kansas", "nm": "New Mexico",
            "ne": "Nebraska", "wv": "West Virginia", "id": "Idaho", "hi": "Hawaii",
            "nh": "New Hampshire", "me": "Maine", "mt": "Montana", "ri": "Rhode Island",
            "de": "Delaware", "sd": "South Dakota", "nd": "North Dakota", "ak": "Alaska",
            "vt": "Vermont", "wy": "Wyoming"
        }

        for state in state_names:
            if state.lower() in user_lower:
                params.states.append(state)
                break

        if not params.states:
            words = user_lower.split()
            for word in words:
                clean_word = word.strip('.,!?;:')
                if clean_word in state_abbrev:
                    params.states.append(state_abbrev[clean_word])
                    break

        if "cheap" in user_lower:
            params.price_max = self.price_stats['p25']
            params.weights['price'] = 0.8
        elif "expensive" in user_lower or "luxury" in user_lower:
            params.price_min = self.price_stats['p75']
            params.weights['price'] = 0.8

        if "hospital" in user_lower:
            params.hospital_required = True
            params.weights['hospital'] = 0.6
            if "near" in user_lower or "nearby" in user_lower:
                params.hospital_distance = "nearby"
                params.weights['hospital'] = 1.0

        if "school" in user_lower:
            params.schools_required = True
            params.weights['schools'] = 0.6

        if "warm" in user_lower:
            params.temperature_min = 17
            params.temperature_max = 25
            params.weights['temperature'] = 0.6

        if "sunny" in user_lower:
            params.weather_preferences.append("sunny")
            params.weights['weather'] = 0.6

        return params

    def auto_adjust_weights(self, params: QueryParameters) -> Dict[str, float]:
        relevant_factors = []

        if params.price_min or params.price_max or params.price_range:
            relevant_factors.append("price")
        if params.hospital_required or params.hospital_distance:
            relevant_factors.append("hospital")
        if params.temperature_preference or params.temperature_min or params.temperature_max:
            relevant_factors.append("temperature")
        if params.weather_preferences:
            relevant_factors.append("weather")
        if params.schools_required or params.schools_distance or params.school_types:
            relevant_factors.append("schools")

        if relevant_factors == ["hospital"]:
            return {"price": 0.0, "hospital": 1.0, "temperature": 0.0, "weather": 0.0, "schools": 0.0}
        if relevant_factors == ["schools"]:
            return {"price": 0.0, "hospital": 0.0, "temperature": 0.0, "weather": 0.0, "schools": 1.0}
        if relevant_factors == ["temperature"]:
            return {"price": 0.0, "hospital": 0.0, "temperature": 1.0, "weather": 0.0, "schools": 0.0}
        if relevant_factors == ["price"]:
            return {"price": 1.0, "hospital": 0.0, "temperature": 0.0, "weather": 0.0, "schools": 0.0}

        if not relevant_factors:
            return {k: 0.2 for k in ["price", "hospital", "temperature", "weather", "schools"]}

        weight = round(1.0 / len(relevant_factors), 2)
        new_weights = {k: (weight if k in relevant_factors else 0.0)
                       for k in ["price", "hospital", "temperature", "weather", "schools"]}
        return new_weights
