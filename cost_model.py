"""Track 2: monthly cost model for Customer Call Reviewer.

Inputs come from two places:
  * per-call averages: KQL query 7 in queries.kql (your own test sessions)
  * unit prices: the Azure pricing calculator, for your region and the standard
    (paid) tiers. Prices are arguments on purpose: the public pricing pages
    render them dynamically, so copy the numbers you see in the calculator and
    screenshot that page for the report.

Example (placeholder prices, NOT real):
  python cost_model.py --seconds-per-call 40 --tts-chars-per-call 190 \
      --text-chars-per-call 600 --stt-per-hour 1.00 --tts-per-million 16 \
      --language-per-1000-records 1.00 --hosting-per-month 13
"""
import argparse
import math

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--calls-per-day", type=float, default=1000)
p.add_argument("--days", type=float, default=30)
p.add_argument("--seconds-per-call", type=float, required=True, help="avg stt_duration_seconds")
p.add_argument("--tts-chars-per-call", type=float, required=True, help="avg tts_char_count")
p.add_argument("--text-chars-per-call", type=float, required=True, help="avg language_text_chars")
p.add_argument("--language-features", type=int, default=4,
               help="Language calls per call: PII, sentiment, key phrases, entities")
p.add_argument("--stt-per-hour", type=float, required=True, help="USD per audio hour (standard real-time)")
p.add_argument("--tts-per-million", type=float, required=True, help="USD per 1M characters (neural)")
p.add_argument("--language-per-1000-records", type=float, required=True,
               help="USD per 1,000 text records (a record is up to 1,000 characters)")
p.add_argument("--hosting-per-month", type=float, default=0.0, help="App Service paid plan, USD/month")
p.add_argument("--monitor-per-month", type=float, default=0.0, help="Log Analytics ingestion beyond free 5 GB")
p.add_argument("--free-stt-hours", type=float, default=5.0)
p.add_argument("--free-tts-chars", type=float, default=500_000)
p.add_argument("--free-language-records", type=float, default=5_000,
               help="F0 monthly free records: confirm on the pricing page")
a = p.parse_args()

calls = a.calls_per_day * a.days
stt_hours = calls * a.seconds_per_call / 3600
tts_chars = calls * a.tts_chars_per_call
records_per_call = math.ceil(a.text_chars_per_call / 1000) * a.language_features
records = calls * records_per_call

rows = [
    ("Speech to text", f"{stt_hours:,.1f} audio hours", max(0, stt_hours - a.free_stt_hours) * a.stt_per_hour),
    ("Text to speech", f"{tts_chars/1e6:,.2f} M characters", max(0, tts_chars - a.free_tts_chars) / 1e6 * a.tts_per_million),
    ("Language (4 features)", f"{records:,.0f} text records", max(0, records - a.free_language_records) / 1000 * a.language_per_1000_records),
    ("App Service hosting", "1 paid plan", a.hosting_per_month),
    ("Monitoring", "ingestion above free 5 GB", a.monitor_per_month),
]
total = sum(r[2] for r in rows)
print(f"\nAssumptions: {a.calls_per_day:,.0f} calls/day x {a.days:.0f} days = {calls:,.0f} calls/month\n")
print(f"{'Stage':<24}{'Monthly usage':<28}{'Cost (USD)':>12}{'Share':>8}")
for name, usage, cost in rows:
    share = f"{cost/total*100:.0f}%" if total else "-"
    print(f"{name:<24}{usage:<28}{cost:>12,.2f}{share:>8}")
print(f"{'TOTAL':<52}{total:>12,.2f}")
top = max(rows, key=lambda r: r[2])
print(f"\nMost expensive stage: {top[0]} (${top[2]:,.2f}/month)")
print(f"Cost per call: ${total/calls:.4f}")
