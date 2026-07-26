"""Generate Helix Robotics support ticket logs.

The narrative documents in data/ are hand-written so their cross-references line
up. Ticket logs are different: they are naturally repetitive, and realism comes
from volume plus consistent reuse of the same sites, people, fault codes, and
version numbers that appear in the hand-written docs.

Deterministic (fixed seed) so re-running does not churn the corpus. Re-run only
if you change the templates.

    python scripts/generate_ticket_logs.py
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

SEED = 20240914
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "support"

# --- Entities shared with the hand-written corpus -------------------------

SITES = [
    # (site name, tier, fleet, weight)
    ("Cardinal Foods — Memphis DC", "Enterprise", "240x HX-200", 26),
    ("Cardinal Foods — Nashville DC", "Enterprise", "88x HX-200, 22x HX-450", 14),
    ("Voss Pharma — Basel DC", "Enterprise", "64x HX-200", 8),
    ("Trellis Home Goods — Columbus DC", "Enterprise", "96x HX-200, 18x HX-450", 13),
    ("Brightline Apparel — Reno DC", "Professional", "54x HX-200", 11),
    ("Meridian Grocers — Louisville DC", "Enterprise", "72x HX-200, 14x HX-450", 10),
    ("Kestrel Distribution — Joliet DC", "Professional", "40x HX-200", 7),
    ("Ardent Supply — Phoenix DC", "Professional", "36x HX-200, 9x HX-450", 6),
    ("Northgate Wholesale — Tacoma DC", "Standard", "28x HX-200", 5),
]

AGENTS = [
    "A. Bello", "R. Whitcombe", "S. Nakamura", "D. Achterberg", "M. Fontaine",
    "T. Osei", "L. Brenner", "C. Villanueva", "P. Deshmukh", "J. Kowalski",
]

# fault code -> (summary templates, typical severity, typical resolution)
FAULTS = [
    (
        "E-207", 22,
        [
            "Multiple units reporting LiDAR occlusion in aisles {a}-{b}",
            "Unit {rid} persistent E-207 on forward safety scanner",
            "E-207 rate elevated across fleet since {when}",
            "Intermittent E-207 clusters during night shift",
        ],
        ["SEV-3", "SEV-3", "SEV-3", "SEV-2"],
        [
            "Sensor windows cleaned by facility staff. Cleared.",
            "Airborne dust. Advised scheduled cleaning cadence; site added to weekly PM.",
            "Scanner replaced by FT-3 technician, post-replacement verification passed.",
            "Resolved by firmware 3.9.0 self-clearing behaviour after 3 clean scans.",
        ],
    ),
    (
        "E-330", 26,
        [
            "WMS sync failure — no tasks arriving since {when}",
            "Site Adapter disconnected from customer WMS",
            "Task state diverging between Fleet OS and customer WMS",
            "E-330 after customer WMS maintenance window",
        ],
        ["SEV-2", "SEV-2", "SEV-2", "SEV-1"],
        [
            "Customer WMS credentials expired. Rotated by customer IT. Excluded from availability.",
            "TLS certificate on WMS endpoint expired. Customer IT reissued.",
            "Customer WMS schema change after upgrade. Sample captured, adapter mapping updated in 4.2.3.",
            "Customer firewall rule change blocked adapter endpoint. Reverted by customer IT.",
        ],
    ),
    (
        "E-115", 12,
        [
            "Unit {rid} e-stop engaged, will not clear",
            "Facility-wide pendant e-stop latched",
            "E-115 on unit {rid} after contact with pallet jack",
        ],
        ["SEV-3", "SEV-2", "SEV-3"],
        [
            "Button mechanically stuck. Replaced by FT-2. Reported to Safety and Compliance per policy.",
            "Pendant reset by facility operator. No fault found.",
            "Physical reset performed after area cleared. Root cause discussed with facility supervisor.",
        ],
    ),
    (
        "E-611", 11,
        [
            "Localization confidence lost, units {rid} and {rid2}",
            "E-611 cluster in aisle {a} after racking change",
            "Unit {rid} cannot localize after cold start",
        ],
        ["SEV-2", "SEV-2", "SEV-3"],
        [
            "Customer moved racking without notifying us. Facility map updated.",
            "Seasonal display staging blocked reference features. Map annotated, customer process agreed.",
            "Repetitive rack geometry defeating scan matching. Resolved by firmware 3.9.0.",
            "Re-localized from Operator Console. Hardware revision A unit, slow cold start (FW-1171).",
        ],
    ),
    (
        "E-502", 8,
        [
            "Battery thermal cutoff on units near loading doors",
            "Unit {rid} E-502 recurring during afternoon shift",
            "E-502 spike fleet-wide during heatwave",
        ],
        ["SEV-3", "SEV-3", "SEV-2"],
        [
            "Ambient exceeded 40 C near dock doors. Facility issue; zone reassignment advised.",
            "Cell voltage deviation high, 3,880 cycles. Battery replaced, cycle-count transfer performed.",
            "Spurious reports on firmware 3.8.0 above 32 C. Resolved by upgrade to 3.8.1.",
        ],
    ),
    (
        "E-704", 7,
        [
            "Unit {rid} drive motor overcurrent, stopped in aisle {a}",
            "Repeated E-704 on unit {rid}",
        ],
        ["SEV-3", "SEV-3"],
        [
            "Shrink wrap wrapped around drive axle. Removed, unit returned to service.",
            "Seized castor wheel replaced by FT-2 technician.",
            "Drive board fault. Unit taken out of service, replacement dispatched.",
        ],
    ),
    (
        "E-118", 9,
        [
            "Payload shift detection halting units at same location",
            "Unit {rid} E-118 in transit from pick face {a}",
            "E-118 rate spike after firmware upgrade",
        ],
        ["SEV-3", "SEV-3", "SEV-2"],
        [
            "Floor expansion joint at reported coordinates. Facility remediation requested.",
            "Totes overloaded beyond rated payload. Customer loading practice addressed.",
            "Expected post-upgrade spike after enabling E-118 on 3.8.0. Reflects pre-existing loading practice.",
        ],
    ),
    (
        "E-401", 8,
        [
            "Charge reservation timeout, units queuing at dock {d}",
            "Unit {rid} could not reach reserved dock in time",
        ],
        ["SEV-3", "SEV-2"],
        [
            "Dock obstructed by manually parked equipment. Cleared with facility staff.",
            "Dock firmware 1.2, handshake failures (FW-1142). Dock updated to 1.4.",
            "Dock provisioning below recommended ratio. Site design review raised with account team.",
        ],
    ),
    (
        "E-820", 6,
        [
            "HX-450 pallet pocket detection failures on inbound pool",
            "Unit {rid} repeated E-820 at staging lane {a}",
        ],
        ["SEV-3", "SEV-3"],
        [
            "Damaged stringers in inbound pallet pool. Raised with customer; pallet quality review agreed.",
            "Shrink-wrap overhang beyond detection tolerance. Wrapping practice adjusted upstream.",
            "Low contrast in staging lane. Fork camera upgrade quoted to customer.",
        ],
    ),
    (
        "E-822", 4,
        [
            "HX-450 load stability fault, unit {rid}",
            "Repeated E-822 on pallets from one supplier",
        ],
        ["SEV-3", "SEV-2"],
        [
            "Asymmetric palletizing upstream of DC. Customer engaged supplier.",
            "Load redistributed, unit returned to service.",
        ],
    ),
    (
        "E-831", 3,
        [
            "HX-450 overhead clearance obstruction in aisle {a}",
        ],
        ["SEV-3"],
        [
            "Sprinkler head lower than surveyed. Facility survey gap; map zone updated.",
            "Improperly stored inventory above lift path. Cleared by facility staff.",
        ],
    ),
]

NON_FAULT = [
    (
        14,
        [
            "Question on availability measurement for {month} invoice",
            "Request for firmware upgrade scheduling",
            "Operator training refresher request",
            "Request for Fleet OS 4.2.3 upgrade window",
            "How do we export task history for internal audit?",
            "Request for site health data without going through TAM",
            "Clarification on dock provisioning ratio for planned expansion",
            "Request for copy of customer-facing incident summary",
        ],
        ["SEV-4"],
        [
            "Answered from Customer FAQ. No action required.",
            "Scheduled with Field Operations for the following maintenance window.",
            "Training session booked with FT-4 trainer.",
            "Explained availability requires dispatch eligibility per revised policy 2024-10-01.",
            "Referred to account team. Self-service dashboard on 2025 roadmap H1.",
        ],
    ),
]


def weighted_pick(rng, items, weight_index):
    total = sum(i[weight_index] for i in items)
    r = rng.uniform(0, total)
    upto = 0.0
    for item in items:
        upto += item[weight_index]
        if upto >= r:
            return item
    return items[-1]


def make_ticket(rng, tid, day, site):
    name, tier, fleet, _ = site
    use_fault = rng.random() < 0.86
    rid = f"HX2-{rng.randint(1, 240):04d}"
    rid2 = f"HX2-{rng.randint(1, 240):04d}"

    if use_fault:
        code, _, summaries, sevs, fixes = weighted_pick(rng, FAULTS, 1)
        # Stacker-only codes require a site with HX-450 units
        if code in ("E-820", "E-822", "E-831") and "HX-450" not in fleet:
            code, _, summaries, sevs, fixes = FAULTS[0]
        if code in ("E-820", "E-822", "E-831"):
            rid = f"HX4-{rng.randint(1, 22):04d}"
        summary = rng.choice(summaries)
        sev = rng.choice(sevs)
        fix = rng.choice(fixes)
    else:
        _, summaries, sevs, fixes = NON_FAULT[0]
        code = "n/a"
        summary = rng.choice(summaries)
        sev = sevs[0]
        fix = rng.choice(fixes)

    summary = (
        summary.replace("{rid}", rid)
        .replace("{rid2}", rid2)
        .replace("{a}", str(rng.randint(1, 28)))
        .replace("{b}", str(rng.randint(29, 46)))
        .replace("{d}", str(rng.randint(1, 12)))
        .replace("{when}", (day - timedelta(days=rng.randint(1, 3))).isoformat())
        .replace("{month}", day.strftime("%B"))
    )

    hours = {"SEV-1": (2, 9), "SEV-2": (1, 14), "SEV-3": (1, 30), "SEV-4": (2, 48)}[sev]
    ttr = rng.randint(*hours)
    if not use_fault:
        # SEV-4 enquiries always originate with the customer
        detection = "Customer report"
    else:
        detection = rng.choices(
            ["Helix automated alerting", "Helix staff observation", "Customer report"],
            weights=[58, 14, 28],
        )[0]

    return {
        "id": f"TKT-{tid}",
        "date": day,
        "site": name,
        "tier": tier,
        "sev": sev,
        "code": code,
        "summary": summary,
        "agent": rng.choice(AGENTS),
        "ttr": ttr,
        "detection": detection,
        "resolution": fix,
    }


def render(t):
    return (
        f"{t['id']}  |  {t['date'].isoformat()}  |  {t['sev']}  |  {t['code']}\n"
        f"  Site        : {t['site']} ({t['tier']})\n"
        f"  Summary     : {t['summary']}\n"
        f"  Agent       : {t['agent']}\n"
        f"  Detection   : {t['detection']}\n"
        f"  Time to res.: {t['ttr']} h\n"
        f"  Resolution  : {t['resolution']}\n"
    )


# --- The INC-2024-017 ticket cluster, written by hand ---------------------
# These are the tickets the narrative documents refer to. They must say exactly
# what the postmortem, runbook, and account record say they say.

INCIDENT_TICKETS = """
TKT-8841  |  2024-09-14  |  SEV-1  |  n/a
  Site        : Cardinal Foods — Memphis DC (Enterprise)
  Summary     : Robots queuing at chargers and not working. Reported by phone by
                Hollis Trent, shift supervisor, at 06:53 CDT. No fault codes
                present on any unit. All robots report healthy.
  Agent       : A. Bello
  Detection   : Customer report
  Time to res.: 7 h
  Resolution  : Opened SEV-2, escalated to SEV-1 at 07:15 and paged engineering
                on-call. 61 of 240 units found holding charging dock
                reservations that were never released, all reporting state of
                charge pinned between 20.0 and 20.5 percent. Reservations
                cleared manually one at a time between 09:20 and 11:40.
                Throughput recovered to baseline 12:59. Root cause: Fleet OS
                4.2.1 strict SoC comparison against firmware 3.8.1 hysteresis
                band, with no expiry lease on dock reservations. See
                INC-2024-017. Detection gap 41 minutes — logged as a monitoring
                defect per policy.

TKT-8842  |  2024-09-14  |  SEV-2  |  E-401
  Site        : Cardinal Foods — Memphis DC (Enterprise)
  Summary     : Charge reservation timeouts on healthy units unable to reach a
                free dock. Secondary effect of TKT-8841 dock queue backup.
  Agent       : A. Bello
  Detection   : Helix automated alerting
  Time to res.: 7 h
  Resolution  : Cleared automatically once the stale reservations from TKT-8841
                were released and the dock queue drained.

TKT-8849  |  2024-09-15  |  SEV-2  |  n/a
  Site        : Cardinal Foods — Nashville DC (Enterprise)
  Summary     : Customer requested urgent confirmation of whether the Memphis
                condition could occur at Nashville.
  Agent       : P. Deshmukh
  Detection   : Customer report
  Time to res.: 4 h
  Resolution  : Nashville HX-200 fleet confirmed on firmware 3.7.2, not exposed.
                HX-450 units run the 2.x firmware line which never adopted the
                SoC hysteresis band and are not affected. Both patches scheduled
                regardless.

TKT-8853  |  2024-09-16  |  SEV-4  |  n/a
  Site        : Voss Pharma — Basel DC (Enterprise)
  Summary     : Anna Kowalczyk (Validation Manager) requested formal assessment
                of INC-2024-017 against the Basel environment.
  Agent       : L. Brenner
  Detection   : Customer report
  Time to res.: 46 h
  Resolution  : Basel runs HX-200 firmware 3.6.x LTS, which predates the
                hysteresis band and is not vulnerable. Written assessment
                provided. Customer change control requires known-relevant fixes
                be applied or formally waived; threshold-crossing logic
                subsequently backported into 3.6.4 (2024-11-19) at their request.

TKT-8861  |  2024-09-18  |  SEV-2  |  n/a
  Site        : Cardinal Foods — Memphis DC (Enterprise)
  Summary     : Deployment of Fleet OS 4.2.2 and HX-200 firmware 3.8.2 across
                240 units.
  Agent       : M. Fontaine
  Detection   : Helix staff observation
  Time to res.: 9 h
  Resolution  : Both patches deployed and verified. Dock reservation leases
                confirmed active. No recurrence. Fleet availability returned to
                baseline.

TKT-8874  |  2024-09-23  |  SEV-4  |  n/a
  Site        : Trellis Home Goods — Columbus DC (Enterprise)
  Summary     : Customer read about the Memphis incident and asked whether their
                site is exposed.
  Agent       : C. Villanueva
  Detection   : Customer report
  Time to res.: 6 h
  Resolution  : Columbus was running Fleet OS 4.2.1 with firmware 3.8.1 —
                exposed. No deadlocked units found, dock ratio provided
                sufficient headroom to mask accumulation. Upgraded to 4.2.2 and
                3.8.2 on 2024-09-25 as a priority. This ticket is the reason we
                now proactively audit version combinations across all sites
                rather than waiting to be asked.
""".strip()


def generate_quarter(rng, label, start, end, start_id, out_path, header_note):
    days = (end - start).days + 1
    tickets = []
    tid = start_id
    for offset in range(days):
        day = start + timedelta(days=offset)
        # weekends are quieter
        base = 2 if day.weekday() >= 5 else 5
        count = max(0, rng.gauss(base, 1.6))
        for _ in range(int(round(count))):
            site = weighted_pick(rng, SITES, 3)
            tickets.append(make_ticket(rng, tid, day, site))
            tid += 1

    lines = [
        "HELIX ROBOTICS — SUPPORT TICKET LOG",
        f"Period: {label}",
        "Owner: Aisha Bello, Director of Customer Support",
        "Classification: Internal — Support and Field Operations",
        "",
        header_note,
        "",
        "Severity definitions and response commitments are in the Support SLA and",
        "Escalation Policy. Fault code diagnostics are in the Troubleshooting Guide.",
        "Detection method is recorded on every ticket; any SEV-1 or SEV-2 detected",
        "by customer report is automatically raised with SRE as a monitoring defect.",
        "",
        "=" * 75,
        "",
    ]

    for t in tickets:
        lines.append(render(t))

    # summary statistics
    by_code = {}
    by_sev = {}
    by_site = {}
    by_detection = {}
    for t in tickets:
        by_code[t["code"]] = by_code.get(t["code"], 0) + 1
        by_sev[t["sev"]] = by_sev.get(t["sev"], 0) + 1
        by_site[t["site"]] = by_site.get(t["site"], 0) + 1
        by_detection[t["detection"]] = by_detection.get(t["detection"], 0) + 1

    lines += ["=" * 75, "", f"QUARTER SUMMARY — {label}", ""]
    lines.append(f"  Total tickets: {len(tickets)}")
    lines.append("")
    lines.append("  By severity:")
    for k in ["SEV-1", "SEV-2", "SEV-3", "SEV-4"]:
        if k in by_sev:
            lines.append(f"    {k}   {by_sev[k]:>4}")
    lines.append("")
    lines.append("  By fault code (top):")
    for k, v in sorted(by_code.items(), key=lambda x: -x[1]):
        lines.append(f"    {k:<6} {v:>4}")
    lines.append("")
    lines.append("  By site:")
    for k, v in sorted(by_site.items(), key=lambda x: -x[1]):
        lines.append(f"    {v:>4}  {k}")
    lines.append("")
    lines.append("  By detection method:")
    for k, v in sorted(by_detection.items(), key=lambda x: -x[1]):
        pct = 100.0 * v / len(tickets)
        lines.append(f"    {v:>4}  ({pct:4.1f}%)  {k}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(tickets), tid


def main():
    rng = random.Random(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    q3_path = DATA_DIR / "ticket_log_2024_q3.txt"
    q4_path = DATA_DIR / "ticket_log_2024_q4.txt"

    n3, next_id = generate_quarter(
        rng,
        "Q3 2024 (July 1 — September 30)",
        date(2024, 7, 1),
        date(2024, 9, 30),
        8100,
        q3_path,
        "NOTE: This quarter contains incident INC-2024-017 (2024-09-14, Cardinal Foods\n"
        "Memphis charge dock reservation deadlock). The ticket cluster for that\n"
        "incident is appended at the end of this log under INCIDENT CLUSTER.",
    )

    # append the hand-written incident cluster to Q3
    with q3_path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 75 + "\n\n")
        fh.write("INCIDENT CLUSTER — INC-2024-017\n\n")
        fh.write(INCIDENT_TICKETS + "\n")

    n4, _ = generate_quarter(
        rng,
        "Q4 2024 (October 1 — December 31)",
        date(2024, 10, 1),
        date(2024, 12, 31),
        next_id,
        q4_path,
        "NOTE: First full quarter operating under the revised Support SLA policy\n"
        "effective 2024-10-01, in which fleet availability requires robots to be\n"
        "eligible for dispatch rather than merely powered, connected, and unfaulted.",
    )

    print(f"Wrote {q3_path.name}: {n3} generated tickets + 6 incident-cluster tickets")
    print(f"Wrote {q4_path.name}: {n4} generated tickets")


if __name__ == "__main__":
    main()
