from typing import List

from realestate.models import PropertyResult, QueryParameters


def format_result_simple(prop: PropertyResult, index: int, params=None) -> str:
    town, state = str(prop.town_state).split("_") if "_" in str(prop.town_state) else (prop.town_state, "")
    price = float(prop.average_price) if prop.average_price else 0

    hospital_5 = prop.hospital_5km
    hospital_20 = prop.hospital_20km
    hospital_100 = prop.hospital_100km

    temperature = prop.temperature
    humidity = prop.humidity
    wind = prop.wind_speed
    sunny = prop.sunny_days
    rainy = prop.rainy_days
    snowy = prop.snowy_days

    pre_10 = prop.prekindergarten_10km
    elem_10 = prop.elementary_10km
    high_10 = prop.high_10km
    pre_20 = prop.prekindergarten_20km
    elem_20 = prop.elementary_20km
    high_20 = prop.high_20km
    pre_100 = prop.prekindergarten_100km
    elem_100 = prop.elementary_100km
    high_100 = prop.high_100km

    html = [f"<div style='margin-bottom:14px;'>"]

    html.append(
        f"<div style='font-weight:700; font-size:1.05rem;'>"
        f"<span style='color:#FF9843;'>{index}.</span> "
        f"<span style='color:#3468C0;'>{town}</span>"
        f"<span style='color:#86A7FC;'>,&nbsp;</span>"
        f"<span style='color:#86A7FC;'>{state}</span>"
        f"</div>"
    )

    html.append("<div style='margin-left:14px;margin-top:6px;font-size:0.92rem;color:#374151;line-height:1.5;'>")

    html.append(f"Average price: <b>${price:,.0f}</b><br>")

    if temperature is not None:
        html.append(f"Average temperature: <b>{temperature:.1f} °C</b><br>")

    html.append("Hospitals nearby:<br>")
    if hospital_5:
        html.append(f"&nbsp;&nbsp;• within 5 km: <b>{hospital_5}</b><br>")
    if hospital_20:
        html.append(f"&nbsp;&nbsp;• within 20 km: <b>{hospital_20}</b><br>")
    if hospital_100:
        html.append(f"&nbsp;&nbsp;• within 100 km: <b>{hospital_100}</b><br>")

    html.append("Schools nearby:<br>")
    html.append(
        f"&nbsp;&nbsp;• within 10 km: "
        f"Prkindergarten <b>{pre_10 or 0}</b> | "
        f"Elementary <b>{elem_10 or 0}</b> | "
        f"High <b>{high_10 or 0}</b><br>"
    )
    html.append(
        f"&nbsp;&nbsp;• within 20 km: "
        f"Prkindergarten <b>{pre_20 or 0}</b> | "
        f"Elementary <b>{elem_20 or 0}</b> | "
        f"High <b>{high_20 or 0}</b><br>"
    )
    html.append(
        f"&nbsp;&nbsp;• within 100 km: "
        f"Prkindergarten <b>{pre_100 or 0}</b> | "
        f"Elementary <b>{elem_100 or 0}</b> | "
        f"High <b>{high_100 or 0}</b><br>"
    )

    html.append("Weather conditions:<br>")
    if humidity is not None:
        html.append(f"&nbsp;&nbsp;• Humidity: <b>{humidity:.0f}%</b><br>")
    if wind is not None:
        html.append(f"&nbsp;&nbsp;• Wind speed: <b>{wind:.1f} m/s</b><br>")
    if sunny is not None:
        html.append(f"&nbsp;&nbsp;• Sunny days: <b>{sunny}</b><br>")
    if rainy is not None:
        html.append(f"&nbsp;&nbsp;• Rainy days: <b>{rainy}</b><br>")
    if snowy is not None:
        html.append(f"&nbsp;&nbsp;• Snowy days: <b>{snowy}</b><br>")

    html.append(f"Overall score: <b>{prop.score:.2f}</b>")

    html.append("</div></div>")
    return "".join(html)


def format_response(
    top: List[PropertyResult],
    fallback_msg: str,
    user_specified_states: bool,
    params: QueryParameters,
    total_results: int,
    top_n: int,
) -> str:
    response = []

    if fallback_msg:
        response.append(fallback_msg)
    else:
        state_info = f" in {', '.join(params.states)}" if user_specified_states else ""
        response.append(f" Found {total_results} locations{state_info}. Top {top_n} according to your criteria:")

    for i, prop in enumerate(top, 1):
        response.append(format_result_simple(prop, i, params))

    return "\n".join(response)
