"""Cost models for on-premise, cloud, and hybrid deployments."""

from dataclasses import dataclass


@dataclass
class YearlyCost:
    year_1: float
    year_2_plus: float
    five_year_tco: float


def on_premise_costs(gpu_tier: str = "mid") -> dict:
    """
    Calculate on-premise GPU server costs.

    gpu_tier: 'low' (2x A6000), 'mid' (2x A100 40GB), 'high' (2x A100 80GB)
    """
    tiers = {
        "low": {"gpu": 10_000, "server": 5_000, "total_hw": 26_000},
        "mid": {"gpu": 30_000, "server": 8_000, "total_hw": 50_000},
        "high": {"gpu": 50_000, "server": 12_000, "total_hw": 77_000},
    }
    t = tiers.get(gpu_tier, tiers["mid"])
    annual_ops = 10_000
    year_1 = t["total_hw"] + annual_ops
    year_2 = annual_ops
    return {
        "hardware_cost": t["total_hw"],
        "annual_operations": annual_ops,
        "yearly": YearlyCost(
            year_1=year_1,
            year_2_plus=year_2,
            five_year_tco=year_1 + year_2 * 4,
        ),
        "data_sovereignty": "Full control",
        "latency": "<100ms",
        "scaling": "Limited by hardware",
        "it_overhead": "High",
    }


def azure_cloud_costs(daily_queries: int = 1000) -> dict:
    """Calculate Azure cloud deployment costs."""
    gpu_vm = 3_500
    app_vm = 140
    postgres = 200
    storage = 5
    network = 50
    monthly = gpu_vm + app_vm + postgres + storage + network
    annual = monthly * 12
    return {
        "monthly_cost": monthly,
        "yearly": YearlyCost(year_1=annual, year_2_plus=annual, five_year_tco=annual * 5),
        "breakdown": {
            "GPU VM (NC24ads A100)": gpu_vm,
            "App VM (D4s v5)": app_vm,
            "PostgreSQL (4 vCores)": postgres,
            "Blob Storage": storage,
            "Networking": network,
        },
        "data_sovereignty": "Azure region",
        "latency": "~200ms",
        "scaling": "Elastic",
        "it_overhead": "Medium",
    }


def aws_cloud_costs(daily_queries: int = 1000) -> dict:
    """Calculate AWS cloud deployment costs."""
    gpu_instance = 1_200
    app_instance = 120
    rds = 150
    s3 = 3
    monthly = gpu_instance + app_instance + rds + s3
    annual = monthly * 12
    return {
        "monthly_cost": monthly,
        "yearly": YearlyCost(year_1=annual, year_2_plus=annual, five_year_tco=annual * 5),
        "breakdown": {
            "GPU (g5.2xlarge)": gpu_instance,
            "App (t3.xlarge)": app_instance,
            "RDS PostgreSQL": rds,
            "S3 Storage": s3,
        },
        "data_sovereignty": "AWS region",
        "latency": "~200ms",
        "scaling": "Elastic",
        "it_overhead": "Medium",
    }


def groq_hybrid_costs(daily_queries: int = 1000) -> dict:
    """Calculate costs using Groq API with cloud backend."""
    avg_tokens_per_query = 2000
    monthly_queries = daily_queries * 30
    monthly_token_cost = (monthly_queries * avg_tokens_per_query / 1_000_000) * 0.69
    cloud_vm = 150
    postgres = 75
    monthly = monthly_token_cost + cloud_vm + postgres
    annual = monthly * 12
    return {
        "monthly_cost": round(monthly, 2),
        "yearly": YearlyCost(
            year_1=round(annual, 2),
            year_2_plus=round(annual, 2),
            five_year_tco=round(annual * 5, 2),
        ),
        "breakdown": {
            "Groq API": round(monthly_token_cost, 2),
            "Cloud VM": cloud_vm,
            "Managed PostgreSQL": postgres,
        },
        "data_sovereignty": "US (Groq servers)",
        "latency": "~500ms",
        "scaling": "Elastic",
        "it_overhead": "Low",
    }


def comparison_table(daily_queries: int = 1000) -> list[dict]:
    """Generate a comparison table across all deployment options."""
    onprem = on_premise_costs("mid")
    azure = azure_cloud_costs(daily_queries)
    aws = aws_cloud_costs(daily_queries)
    hybrid = groq_hybrid_costs(daily_queries)

    return [
        {
            "option": "On-Premise GPU",
            "year_1": onprem["yearly"].year_1,
            "year_2_plus": onprem["yearly"].year_2_plus,
            "five_year_tco": onprem["yearly"].five_year_tco,
            "data_sovereignty": onprem["data_sovereignty"],
            "latency": onprem["latency"],
            "scaling": onprem["scaling"],
        },
        {
            "option": "Azure Cloud",
            "year_1": azure["yearly"].year_1,
            "year_2_plus": azure["yearly"].year_2_plus,
            "five_year_tco": azure["yearly"].five_year_tco,
            "data_sovereignty": azure["data_sovereignty"],
            "latency": azure["latency"],
            "scaling": azure["scaling"],
        },
        {
            "option": "AWS Cloud",
            "year_1": aws["yearly"].year_1,
            "year_2_plus": aws["yearly"].year_2_plus,
            "five_year_tco": aws["yearly"].five_year_tco,
            "data_sovereignty": aws["data_sovereignty"],
            "latency": aws["latency"],
            "scaling": aws["scaling"],
        },
        {
            "option": "Groq API Hybrid",
            "year_1": hybrid["yearly"].year_1,
            "year_2_plus": hybrid["yearly"].year_2_plus,
            "five_year_tco": hybrid["yearly"].five_year_tco,
            "data_sovereignty": hybrid["data_sovereignty"],
            "latency": hybrid["latency"],
            "scaling": hybrid["scaling"],
        },
    ]
