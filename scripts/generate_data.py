"""
generate_data.py
Builds a synthetic CRM dataset for "Acceldata" (enterprise data observability,
selling into the US and Europe). Output: 8 CSV files in the data/ folder.

The data is deliberately messy (duplicates, missing values, bad titles,
stage-skipping deals, slipped close dates) so later projects have real
problems to fix.

Run from the repo root:
    python3 scripts/generate_data.py
"""

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# 0. Settings: change these numbers to make the dataset bigger or smaller
# ---------------------------------------------------------------------------
SEED = 42                      # same seed = same data every run
N_ACCOUNTS = 2000
N_CONTACTS = 6000              # before duplicates are added
DUPLICATE_RATE = 0.15          # 15% extra duplicate contacts
N_LEADS = 1500
N_OPPS = 400
TODAY = date(2026, 10, 8)
START = date(2025, 4, 1)       # 6 quarters back: Q2 2025 to Q3 2026

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
fake = Faker(["en_US", "en_GB", "de_DE", "nl_NL", "sv_SE"])
Faker.seed(SEED)


def rand_date(start, end):
    """A random date between start and end (inclusive)."""
    return start + timedelta(days=random.randint(0, (end - start).days))


# ---------------------------------------------------------------------------
# 1. Reference lists (the "business rules" of the fake company)
# ---------------------------------------------------------------------------
COUNTRIES = {  # country: (sub_region, sales_region, weight)
    "United States": ("NA", "North America", 45),
    "Canada": ("NA", "North America", 5),
    "United Kingdom": ("UK&I", "UK&I", 15),
    "Ireland": ("UK&I", "UK&I", 3),
    "Germany": ("DACH", "Continental Europe", 10),
    "Switzerland": ("DACH", "Continental Europe", 4),
    "Austria": ("DACH", "Continental Europe", 2),
    "Netherlands": ("Benelux", "Continental Europe", 5),
    "Belgium": ("Benelux", "Continental Europe", 3),
    "Sweden": ("Nordics", "Continental Europe", 4),
    "Denmark": ("Nordics", "Continental Europe", 2),
    "Norway": ("Nordics", "Continental Europe", 2),
}
EU_COUNTRIES = [c for c, v in COUNTRIES.items() if v[0] != "NA"]

INDUSTRIES = ["Banking", "Insurance", "Telecom", "Retail",
              "Healthcare", "Manufacturing", "CPG"]
DATA_STACK = ["Snowflake", "Databricks", "BigQuery", "Hadoop/Cloudera",
              "dbt", "Airflow", "Redshift"]
COMPETITORS = ["Monte Carlo", "Bigeye", "Datadog", "In-house", "None"]

PERSONAS = {  # persona: list of clean job titles
    "Economic Buyer": ["Chief Data Officer", "VP Data", "VP Data & Analytics",
                       "Chief Information Officer"],
    "Champion": ["Head of Data Engineering", "Director of Data Platform",
                 "Head of Data Platform", "Director, Data Engineering"],
    "User": ["Senior Data Engineer", "Data Engineering Manager",
             "Analytics Engineer", "Staff Data Engineer"],
    "Blocker": ["CISO", "Head of Procurement", "IT Security Manager"],
}
BAD_TITLES = ["x", "N/A", "intern", "data guy", "-", "ceo ", "DATA ENG",
              "Manager", "test"]

STAGES = ["Discovery", "Technical Validation", "POC", "Business Case",
          "Negotiation/Procurement"]
STAGE_WIN_PROB = {"Discovery": 0.10, "Technical Validation": 0.20,
                  "POC": 0.40, "Business Case": 0.60,
                  "Negotiation/Procurement": 0.80}
LEAD_SOURCES = ["SDR Outbound", "Event", "Webinar", "Paid Search",
                "Content Download", "Partner", "Free Trial", "Referral"]


# ---------------------------------------------------------------------------
# 2. users.csv: 3 regions x (1 manager, 2 AEs, 2 SDRs, 1 SE) = 18 people
# ---------------------------------------------------------------------------
users = []
uid = 1
for region in ["North America", "UK&I", "Continental Europe"]:
    manager_id = f"U{uid:03d}"
    for role in ["Sales Manager", "AE", "AE", "SDR", "SDR", "SE"]:
        name = fake.name()
        users.append({
            "user_id": f"U{uid:03d}",
            "full_name": name,
            "email": name.lower().replace(" ", ".") + "@acceldata-demo.io",
            "role": role,
            "sales_region": region,
            "manager_id": None if role == "Sales Manager" else manager_id,
            "start_date": rand_date(date(2022, 1, 1), date(2025, 9, 1)),
            "is_active": True,
        })
        uid += 1
users_df = pd.DataFrame(users)
# One AE left the company: realistic for ownership clean-up later
users_df.loc[users_df.index[7], "is_active"] = False

def reps(role, region):
    """List of user_ids with a given role in a given sales region."""
    m = (users_df.role == role) & (users_df.sales_region == region)
    return users_df.loc[m, "user_id"].tolist()


# ---------------------------------------------------------------------------
# 3. accounts.csv
# ---------------------------------------------------------------------------
country_names = list(COUNTRIES)
country_weights = [v[2] for v in COUNTRIES.values()]

accounts = []
for i in range(1, N_ACCOUNTS + 1):
    country = random.choices(country_names, weights=country_weights)[0]
    sub_region, sales_region, _ = COUNTRIES[country]
    employees = int(np.random.lognormal(mean=7.5, sigma=1.0))  # skewed sizes
    employees = max(250, min(employees, 250000))
    segment = ("Strategic" if employees >= 10000 else
               "Enterprise" if employees >= 1000 else "Mid-market")
    name = fake.company().replace(",", "")
    domain = (name.lower().split()[0].strip(".-") + ".com")
    accounts.append({
        "account_id": f"A{i:05d}",
        "account_name": name,
        "domain": domain,
        "industry": random.choice(INDUSTRIES),
        "employees": employees,
        "annual_revenue_usd": employees * random.randint(150_000, 400_000),
        "country": country,
        "sub_region": sub_region,
        "sales_region": sales_region,
        "segment": segment,
        "data_stack": ";".join(random.sample(DATA_STACK, random.randint(1, 3))),
        "account_owner_id": random.choice(reps("AE", sales_region)),
        "account_type": random.choices(["Prospect", "Customer", "Churned"],
                                       weights=[88, 10, 2])[0],
        "is_target_account": random.random() < 0.15,
        "created_date": rand_date(date(2023, 1, 1), TODAY),
    })
accounts_df = pd.DataFrame(accounts)

# MESS: 8% of accounts have no industry
accounts_df.loc[accounts_df.sample(frac=0.08, random_state=1).index,
                "industry"] = None


# ---------------------------------------------------------------------------
# 4. contacts.csv (+15% duplicates)
# ---------------------------------------------------------------------------
contacts = []
for i in range(1, N_CONTACTS + 1):
    acc = accounts_df.sample(1, random_state=SEED + i).iloc[0]
    persona = random.choices(list(PERSONAS), weights=[15, 25, 50, 10])[0]
    first, last = fake.first_name(), fake.last_name()
    contacts.append({
        "contact_id": f"C{i:06d}",
        "account_id": acc.account_id,
        "first_name": first,
        "last_name": last,
        "email": f"{first}.{last}@{acc.domain}".lower(),
        "job_title": random.choice(PERSONAS[persona]),
        "persona": persona,
        "country": acc.country,
        "lawful_basis": ("Legitimate Interest" if acc.country in EU_COUNTRIES
                         else None),
        "email_opt_out": random.random() < 0.04,
        "lead_source": random.choice(LEAD_SOURCES),
        "created_date": rand_date(acc.created_date, TODAY),
    })
contacts_df = pd.DataFrame(contacts)

# MESS: 10% bad job titles, and persona unknown for those rows
bad = contacts_df.sample(frac=0.10, random_state=2).index
contacts_df.loc[bad, "job_title"] = [random.choice(BAD_TITLES) for _ in bad]
contacts_df.loc[bad, "persona"] = None

# MESS: 3% of EU contacts are missing lawful_basis (a GDPR gap to find)
eu = contacts_df[contacts_df.country.isin(EU_COUNTRIES)]
contacts_df.loc[eu.sample(frac=0.03, random_state=3).index,
                "lawful_basis"] = None

# MESS: 15% duplicates with small variations (case, nickname, extra space)
dups = contacts_df.sample(frac=DUPLICATE_RATE, random_state=4).copy()
dups["contact_id"] = [f"C{N_CONTACTS + k + 1:06d}" for k in range(len(dups))]
variation = np.random.choice(["upper_email", "short_name", "space", "same"],
                             size=len(dups))
dups.loc[variation == "upper_email", "email"] = \
    dups.loc[variation == "upper_email", "email"].str.upper()
dups.loc[variation == "short_name", "first_name"] = \
    dups.loc[variation == "short_name", "first_name"].str[:3]
dups.loc[variation == "space", "last_name"] = \
    dups.loc[variation == "space", "last_name"] + " "
dups["created_date"] = [rand_date(d, TODAY) for d in dups.created_date]
contacts_df = pd.concat([contacts_df, dups], ignore_index=True)


# ---------------------------------------------------------------------------
# 5. campaigns.csv
# ---------------------------------------------------------------------------
CAMPAIGN_PLAN = [  # (type, name pattern, budget range)
    ("Event", "dbt Coalesce {y}", (40000, 90000)),
    ("Event", "Snowflake Summit {y}", (60000, 120000)),
    ("Event", "Data + AI Summit {y}", (50000, 110000)),
    ("Webinar", "Data Reliability Webinar {q}", (3000, 8000)),
    ("Paid", "LinkedIn Ads - Data Leaders {q}", (15000, 40000)),
    ("Content", "State of Data Observability Report {q}", (5000, 15000)),
    ("Outbound", "SDR Hadoop Migration Play {q}", (0, 0)),
    ("Partner", "Cloud Marketplace Co-sell {q}", (10000, 25000)),
]
campaigns = []
cid = 1
q_starts = pd.date_range(START, TODAY, freq="QS").date
for q_start in q_starts:
    q_label = f"Q{(q_start.month - 1) // 3 + 1} {q_start.year}"
    for ctype, pattern, (lo, hi) in CAMPAIGN_PLAN:
        if ctype == "Event" and random.random() < 0.6:
            continue  # events don't happen every quarter
        start = rand_date(q_start, q_start + timedelta(days=60))
        campaigns.append({
            "campaign_id": f"CMP{cid:03d}",
            "campaign_name": pattern.format(y=q_start.year, q=q_label),
            "campaign_type": ctype,
            "sales_region": random.choice(["Global", "North America",
                                           "UK&I", "Continental Europe"]),
            "start_date": start,
            "end_date": start + timedelta(days=random.randint(1, 30)),
            "budget_usd": random.randint(lo, hi) if hi else 0,
        })
        cid += 1
campaigns_df = pd.DataFrame(campaigns)


# ---------------------------------------------------------------------------
# 6. leads.csv (inbound + outbound leads, some converted to contacts)
# ---------------------------------------------------------------------------
leads = []
for i in range(1, N_LEADS + 1):
    created = rand_date(START, TODAY)
    # 60% of leads come from companies already in accounts.csv
    from_existing = random.random() < 0.60
    if from_existing:
        acc = accounts_df.sample(1, random_state=SEED * 3 + i).iloc[0]
        country, company, domain = acc.country, acc.account_name, acc.domain
        # MESS: 30% of them type the company name differently
        if random.random() < 0.30:
            company = company + random.choice([" Inc", " GmbH", " Ltd", " Group"])
    else:
        country = random.choices(country_names, weights=country_weights)[0]
        company = fake.company().replace(",", "")
        domain = company.lower().split()[0].strip(".-") + ".com"
    sales_region = COUNTRIES[country][1]
    first, last = fake.first_name(), fake.last_name()
    # MESS: 12% of leads use personal emails (hard to match to accounts)
    if random.random() < 0.12:
        domain = random.choice(["gmail.com", "outlook.com", "yahoo.com"])
    persona = random.choice(list(PERSONAS))

    # Funnel: Lead -> MQL (45%) -> SQL (40% of MQL) -> Converted (60% of SQL)
    status, mql_date, sql_date = "New", None, None
    if random.random() < 0.45:
        status, mql_date = "MQL", created + timedelta(days=random.randint(0, 20))
        if random.random() < 0.40:
            status = "SQL"
            sql_date = mql_date + timedelta(days=random.randint(1, 14))
            if random.random() < 0.60:
                status = "Converted"
    elif random.random() < 0.3:
        status = "Disqualified"
    # Leads created in the last few days can't be far down the funnel
    if mql_date and mql_date > TODAY:
        status, mql_date, sql_date = "New", None, None
    if sql_date and sql_date > TODAY:
        status, sql_date = "MQL", None

    camp = campaigns_df.sample(1, random_state=SEED + i).iloc[0]
    leads.append({
        "lead_id": f"L{i:05d}",
        "first_name": first,
        "last_name": last,
        "email": f"{first}.{last}@{domain}".lower(),
        "company": company,
        "job_title": random.choice(PERSONAS[persona]),
        "country": country,
        "sales_region": sales_region,
        "lead_source": random.choice(LEAD_SOURCES),
        "campaign_id": camp.campaign_id,
        "lead_status": status,
        "lead_score": random.randint(0, 100),
        "owner_id": random.choice(reps("SDR", sales_region)),
        "created_date": created,
        "mql_date": mql_date,
        "sql_date": sql_date,
        "converted_account_id": (acc.account_id
                                 if status == "Converted" and from_existing
                                 else None),
        "converted_contact_id": None,  # filled in below
    })
leads_df = pd.DataFrame(leads)

# Lead conversion: each Converted lead becomes a contact. If its company is
# not in accounts.csv yet, a new account is created for it first.
TITLE_TO_PERSONA = {t: p for p, titles in PERSONAS.items() for t in titles}
new_accounts, new_contacts = [], []
for idx, lead in leads_df[leads_df.lead_status == "Converted"].iterrows():
    account_id = lead.converted_account_id
    if pd.isna(account_id):  # net-new company -> create the account
        account_id = f"A{N_ACCOUNTS + len(new_accounts) + 1:05d}"
        employees = random.randint(250, 5000)
        new_accounts.append({
            "account_id": account_id,
            "account_name": lead.company,
            "domain": lead.company.lower().split()[0].strip(".-") + ".com",
            "industry": random.choice(INDUSTRIES),
            "employees": employees,
            "annual_revenue_usd": employees * random.randint(150_000, 400_000),
            "country": lead.country,
            "sub_region": COUNTRIES[lead.country][0],
            "sales_region": lead.sales_region,
            "segment": "Enterprise" if employees >= 1000 else "Mid-market",
            "data_stack": ";".join(random.sample(DATA_STACK, 2)),
            "account_owner_id": random.choice(reps("AE", lead.sales_region)),
            "account_type": "Prospect",
            "is_target_account": False,
            "created_date": lead.sql_date,
        })
    contact_id = f"C{len(contacts_df) + len(new_contacts) + 1:06d}"
    new_contacts.append({
        "contact_id": contact_id,
        "account_id": account_id,
        "first_name": lead.first_name,
        "last_name": lead.last_name,
        "email": lead.email,
        "job_title": lead.job_title,
        "persona": TITLE_TO_PERSONA.get(lead.job_title),
        "country": lead.country,
        "lawful_basis": ("Legitimate Interest" if lead.country in EU_COUNTRIES
                         else None),
        "email_opt_out": False,
        "lead_source": lead.lead_source,
        "created_date": lead.sql_date,
    })
    leads_df.loc[idx, "converted_account_id"] = account_id
    leads_df.loc[idx, "converted_contact_id"] = contact_id

accounts_df = pd.concat([accounts_df, pd.DataFrame(new_accounts)],
                        ignore_index=True)
contacts_df = pd.concat([contacts_df, pd.DataFrame(new_contacts)],
                        ignore_index=True)
# MESS: 5% of leads have no owner (routing gap)
leads_df.loc[leads_df.sample(frac=0.05, random_state=5).index,
             "owner_id"] = None


# ---------------------------------------------------------------------------
# 7. opportunities.csv + opportunity_history.csv
# ---------------------------------------------------------------------------
opps, history = [], []
for i in range(1, N_OPPS + 1):
    acc = accounts_df.sample(1, random_state=SEED * 7 + i).iloc[0]
    region = acc.sales_region
    owner = acc.account_owner_id
    created = rand_date(START, TODAY - timedelta(days=10))
    opp_type = random.choices(["New Logo", "Expansion", "Renewal"],
                              weights=[70, 20, 10])[0]
    base = {"Strategic": 250000, "Enterprise": 120000, "Mid-market": 60000}
    amount = int(base[acc.segment] * random.uniform(0.5, 1.6) / 1000) * 1000

    # How far the deal gets. Deals created long ago are more likely closed.
    age_days = (TODAY - created).days
    path = list(STAGES)
    # MESS: 12% of deals skip a middle stage (e.g. Discovery -> POC)
    if random.random() < 0.12:
        path.remove(random.choice(STAGES[1:4]))

    furthest = random.randint(1, len(path))
    closed = age_days > 150 or (age_days > 60 and random.random() < 0.5)
    won = closed and furthest >= len(path) - 1 and random.random() < 0.55

    # Close dates: original plan, then pushes (MESS: slipped close dates)
    original_close = created + timedelta(days=random.randint(90, 240))
    pushes = np.random.choice([0, 1, 2, 3], p=[0.55, 0.25, 0.13, 0.07])
    close_date = original_close + timedelta(days=int(pushes) * 30)

    # Walk the stages and write one history row per change
    d = created
    stage_list = path[:furthest]
    for s_idx, stage in enumerate(stage_list):
        history.append({
            "opportunity_id": f"O{i:04d}", "stage": stage,
            "amount_usd": amount, "close_date": original_close,
            "changed_date": d, "changed_by": owner})
        d = d + timedelta(days=random.randint(10, 45))
        if d > TODAY:
            break
    for p in range(int(pushes)):  # a history row for every close-date push
        push_day = min(d, TODAY)
        history.append({
            "opportunity_id": f"O{i:04d}", "stage": stage,
            "amount_usd": amount,
            "close_date": original_close + timedelta(days=30 * (p + 1)),
            "changed_date": push_day, "changed_by": owner})

    if closed:
        final_stage = "Closed Won" if won else "Closed Lost"
        close_date = min(close_date, TODAY - timedelta(days=1))
        close_date = max(close_date, created + timedelta(days=30))
        history.append({
            "opportunity_id": f"O{i:04d}", "stage": final_stage,
            "amount_usd": amount, "close_date": close_date,
            "changed_date": close_date, "changed_by": owner})
    else:
        final_stage = stage

    forecast = ("Closed" if closed else
                "Commit" if final_stage == "Negotiation/Procurement" else
                "Best Case" if final_stage in ("POC", "Business Case") else
                "Pipeline")
    meddicc = random.randint(2, 7) + 2 * STAGES.index(stage) if stage in STAGES else 0
    contacts_here = contacts_df[contacts_df.account_id == acc.account_id]

    opps.append({
        "opportunity_id": f"O{i:04d}",
        "opportunity_name": f"{acc.account_name} - {opp_type}",
        "account_id": acc.account_id,
        "owner_id": owner,
        "se_id": random.choice(reps("SE", region)),
        "opportunity_type": opp_type,
        "stage": final_stage,
        "forecast_category": forecast,
        "amount_usd": amount,
        "created_date": created,
        "original_close_date": original_close,
        "close_date": close_date,
        "close_date_push_count": int(pushes),
        "is_closed": closed,
        "is_won": won,
        "lead_source": random.choice(LEAD_SOURCES),
        "primary_campaign_id": random.choice(campaigns_df.campaign_id.tolist()
                                             + [None] * 10),
        "primary_competitor": random.choice(COMPETITORS),
        "meddicc_score": min(meddicc, 16),
        "economic_buyer_contact_id": (contacts_here.contact_id.iloc[0]
                                      if len(contacts_here) and random.random() < 0.6
                                      else None),
        "champion_contact_id": (contacts_here.contact_id.iloc[-1]
                                if len(contacts_here) and random.random() < 0.7
                                else None),
        "loss_reason": (random.choice(["No budget", "Chose competitor",
                                       "Built in-house", "No decision",
                                       "Timing"]) if closed and not won else None),
    })
opps_df = pd.DataFrame(opps)
history_df = pd.DataFrame(history).sort_values(["opportunity_id", "changed_date"])
history_df.insert(0, "history_id", [f"H{k:05d}" for k in range(1, len(history_df) + 1)])

# MESS: 3% of open deals have a close date in the past (stale pipeline)
open_idx = opps_df[~opps_df.is_closed].sample(frac=0.03, random_state=6).index
opps_df.loc[open_idx, "close_date"] = [rand_date(START, TODAY - timedelta(days=15))
                                       for _ in open_idx]


# ---------------------------------------------------------------------------
# 8. activities.csv (emails, calls, meetings, LinkedIn touches)
# ---------------------------------------------------------------------------
activities = []
aid = 1
# Activities on opportunities: open deals get fewer recent touches sometimes
for opp in opps_df.itertuples():
    n = random.randint(4, 25)
    contacts_here = contacts_df[contacts_df.account_id == opp.account_id]
    end = min(opp.close_date, TODAY) if opp.is_closed else TODAY
    # MESS: 15% of open deals have gone quiet (no activity in last 21+ days)
    if not opp.is_closed and random.random() < 0.15:
        end = TODAY - timedelta(days=random.randint(21, 60))
    for _ in range(n):
        activities.append({
            "activity_id": f"ACT{aid:06d}",
            "activity_type": random.choices(["Email", "Call", "Meeting", "LinkedIn"],
                                            weights=[45, 25, 20, 10])[0],
            "owner_id": random.choice([opp.owner_id, opp.se_id]),
            "account_id": opp.account_id,
            "contact_id": (contacts_here.sample(1).contact_id.iloc[0]
                           if len(contacts_here) else None),
            "opportunity_id": opp.opportunity_id,
            "lead_id": None,
            "activity_date": rand_date(opp.created_date, max(end, opp.created_date)),
            "outcome": random.choice(["Completed", "No answer", "Replied",
                                      "Meeting held", "Bounced"]),
        })
        aid += 1
# Prospecting activities by SDRs on leads (not tied to a deal)
for lead in leads_df.sample(frac=0.6, random_state=7).itertuples():
    for _ in range(random.randint(1, 6)):
        activities.append({
            "activity_id": f"ACT{aid:06d}",
            "activity_type": random.choice(["Email", "Call", "LinkedIn"]),
            "owner_id": lead.owner_id,
            "account_id": None,
            "contact_id": None,
            "opportunity_id": None,
            "lead_id": lead.lead_id,
            "activity_date": rand_date(lead.created_date,
                                       max(lead.created_date, TODAY)),
            "outcome": random.choice(["Completed", "No answer", "Replied", "Bounced"]),
        })
        aid += 1
activities_df = pd.DataFrame(activities)


# ---------------------------------------------------------------------------
# 9. Save everything and print a quick summary
# ---------------------------------------------------------------------------
tables = {
    "users": users_df, "accounts": accounts_df, "contacts": contacts_df,
    "campaigns": campaigns_df, "leads": leads_df, "opportunities": opps_df,
    "opportunity_history": history_df, "activities": activities_df,
}
for name, df in tables.items():
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name:<22}{len(df):>7,} rows  ->  data/{name}.csv")
