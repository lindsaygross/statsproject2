#!/usr/bin/env python3
"""
Parse Apple Health export XML and compute steps per day + average.
Usage:
    python apple_steps.py /path/to/export.xml [--source "iPhone"] [--min-date YYYY-MM-DD] [--max-date YYYY-MM-DD]
"""
import sys
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, date
from dateutil import parser as dateparser   # pip install python-dateutil
import csv
import collections

def iter_records(xml_path, tagname="Record"):
    # Use iterparse to avoid loading whole file into memory
    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag == tagname:
            yield elem
            elem.clear()

def parse_steps(xml_path, filter_source=None, min_date=None, max_date=None):
    totals = collections.defaultdict(float)  # date -> steps (float, but steps are integers)
    count_seen = 0
    for rec in iter_records(xml_path, tagname="Record"):
        rtype = rec.get("type")
        if rtype != "HKQuantityTypeIdentifierStepCount":
            continue
        count_seen += 1
        # attributes commonly present: value, startDate, endDate, sourceName, sourceVersion
        val = rec.get("value")
        if val is None:
            continue
        try:
            steps = float(val)
        except:
            continue
        src = rec.get("sourceName", "")
        if filter_source and filter_source != src:
            continue
        # pick a timestamp to bucket by day. Using startDate is common.
        sdate = rec.get("startDate")
        if sdate is None:
            sdate = rec.get("endDate")
        if sdate is None:
            continue
        try:
            dt = dateparser.parse(sdate)  # handles timezone offsets like "2018-07-10 13:00:00 -0700"
        except Exception as e:
            # fallback: try simple split
            try:
                dt = datetime.fromisoformat(sdate)
            except:
                continue
        day = dt.date()
        if min_date and day < min_date:
            continue
        if max_date and day > max_date:
            continue
        totals[day] += steps

    return totals, count_seen

def write_csv(totals, out_path="steps_per_day.csv"):
    rows = sorted(totals.items())
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "steps"])
        for d, s in rows:
            writer.writerow([d.isoformat(), int(round(s))])
    print(f"Wrote {len(rows)} rows to {out_path}")

def pretty_print(totals):
    if not totals:
        print("No step records found after filters.")
        return
    rows = sorted(totals.items())
    total_steps = sum(totals.values())
    days = len(rows)
    avg = total_steps / days
    print(f"Days with data: {days}")
    print(f"Total steps across those days: {int(round(total_steps)):,}")
    print(f"Average steps per day (over days with data): {int(round(avg)):,}")
    print()
    print("Sample (date, steps):")
    for d, s in rows[:10]:
        print(f"  {d.isoformat()}  {int(round(s)):,}")
    if days > 10:
        print("  ...")
    # return numeric summary too
    return {"days": days, "total_steps": total_steps, "avg": avg}

def parse_args():
    p = argparse.ArgumentParser(description="Aggregate Apple Health step counts by day.")
    p.add_argument("xml", help="Path to export.xml (Apple Health export)")
    p.add_argument("--source", help='Filter by sourceName (e.g. "iPhone", "Apple Watch"). If omitted, sums all sources (may double-count).')
    p.add_argument("--min-date", help="Earliest date (YYYY-MM-DD) to include")
    p.add_argument("--max-date", help="Latest date (YYYY-MM-DD) to include")
    p.add_argument("--csv", action="store_true", help="Write steps_per_day.csv")
    return p.parse_args()

def iso_to_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()

def main():
    args = parse_args()
    min_d = iso_to_date(args.min_date) if args.min_date else None
    max_d = iso_to_date(args.max_date) if args.max_date else None
    print("Parsing (this may take a few seconds for large exports)...")
    totals, seen = parse_steps(args.xml, filter_source=args.source, min_date=min_d, max_date=max_d)
    print(f"Total Record elements scanned (all types): {seen}, {len(totals)}")
    summary = pretty_print(totals)
    if args.csv:
        write_csv(totals)
    # also print recommendations if user didn't filter by source
    if not args.source:
        print("\nNote: you did not filter by source. Apple Health export often contains step records from multiple sources (iPhone, Apple Watch, third-party apps). Summing all sources can double-count overlapping data. To avoid that, re-run with --source \"iPhone\" or similar.")
    return summary

if __name__ == "__main__":
    main()
