# Data Dictionary

## Raw Data

### PJM LMP

| Column | Type | Description |
|--------|------|-------------|
| datetime_begin | datetime64[ns, UTC] | Interval start hour |
| lmp / total_lmp | float | Locational marginal price ($/MWh) |
| node_id | str | PJM node or zone identifier |

### Weather (Open-Meteo)

| Column | Type | Description |
|--------|------|-------------|
| datetime | datetime64[ns, UTC] | Hour |
| temperature_2m | float | °C |
| relative_humidity_2m | float | % |
| cloud_cover | float | % |
| wind_speed_10m | float | km/h |
| precipitation | float | mm |

## Processed / Feature Data

| Column | Type | Description |
|--------|------|-------------|
| datetime | datetime64[ns, UTC] | Hour (UTC) |
| target | float | LMP to predict ($/MWh) |
| lag_1, lag_2, lag_24, lag_48, lag_168 | float | Lagged prices (h) |
| roll_{w}_mean, roll_{w}_std, roll_{w}_min, roll_{w}_max | float | Rolling stats (w=6,24,168) |
| roll_{w}_q25, roll_{w}_q50, roll_{w}_q75 | float | Rolling quantiles |
| hour | int | 0–23 |
| dow | int | Day of week 0–6 |
| month | int | 1–12 |
| is_weekend | int | 0/1 |
| is_holiday | int | 0/1 |
| price_change_1, price_change_24 | float | 1h and 24h diff |
| volatility_24 | float | 24h rolling std |

## Units

- Prices: $/MWh (USD per megawatt-hour)
- Timestamps: UTC (ISO 8601)
