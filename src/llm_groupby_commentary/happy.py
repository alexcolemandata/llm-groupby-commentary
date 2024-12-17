"""World happiness data"""

from pathlib import Path

import polars as pl
import re
import pandera.polars as pa
from pandera.polars import check_io

from polars.datatypes import Int32
import polars.selectors as cs

from llm_groupby_commentary import DATA_DIR

SOURCE_DIR = DATA_DIR / "world_happiness"

schema_raw = pa.DataFrameSchema(
    {
        "year": pa.Column(Int32),
        "country": pa.Column(str),
        "region": pa.Column(str, required=False),
        "happiness_rank": pa.Column(int),
        "happiness_score": pa.Column(float),
        "gdp_per_capita": pa.Column(float),
        "healthy_life_expectancy": pa.Column(float),
        "score_family_and_social_support": pa.Column(float),
        "score_freedom": pa.Column(float),
        "score_perception_of_corruption": pa.Column(float, nullable=True),
        "score_generosity": pa.Column(float),
    },
    description="initial data does not have region defined for all years",
)

schema_parsed = schema_raw.update_column(
    "region", required=True, description="filled region from other years"
)

schema_stats = pa.DataFrameSchema(
    {
        "year": pa.Column(Int32),
        "happiness_quartile": pa.Column(str),
        "num_countries": pa.Column(int),
        "mean_happiness_score": pa.Column(float),
        "min_happiness_score": pa.Column(float),
        "max_happiness_score": pa.Column(float),
        "mean_gdp_per_capita": pa.Column(float),
        "min_gdp_per_capita": pa.Column(float),
        "max_gdp_per_capita": pa.Column(float),
        "mean_healthy_life_expectancy": pa.Column(float),
        "min_healthy_life_expectancy": pa.Column(float),
        "max_healthy_life_expectancy": pa.Column(float),
        "mean_social_support": pa.Column(float),
        "min_social_support": pa.Column(float),
        "max_social_support": pa.Column(float),
        "mean_freedom_score": pa.Column(float),
        "min_freedom_score": pa.Column(float),
        "max_freedom_score": pa.Column(float),
        "mean_perception_of_corruption": pa.Column(float),
        "min_perception_of_corruption": pa.Column(float),
        "max_perception_of_corruption": pa.Column(float),
    },
    description="Summary statistics for global and quartiles per year",
    coerce=True,
)


@check_io(out=schema_raw)
def read_data_file(path: Path) -> pl.LazyFrame:
    result = (
        pl.read_csv(path)
        .rename(
            lambda col: re.sub(pattern=r"[ .)(]+", repl="_", string=col.lower()).strip(
                "_"
            ),
        )
        .rename(
            {
                "country_or_region": "country",
                "health_life_expectancy": "healthy_life_expectancy",
                "generosity": "score_generosity",
                "trust_government_corruption": "score_perception_of_corruption",
                "perceptions_of_corruption": "score_perception_of_corruption",
                "overall_rank": "happiness_rank",
                "score": "happiness_score",
                "freedom": "score_freedom",
                "freedom_to_make_life_choices": "score_freedom",
                "economy_gdp_per_capita": "gdp_per_capita",
                "family": "score_family_and_social_support",
                "social_support": "score_family_and_social_support",
            },
            strict=False,
        )
        .with_columns(
            pl.col("score_perception_of_corruption").cast(pl.Float64, strict=False),
            year=pl.lit(int(path.stem)),
        )
        # could skip dropping, but would have to concat using how='diagonal_relaxed'
        # and would result in a lot of nulls for not much value
        # keeping things simple for now
        .drop(
            [
                "whisker_high",
                "whisker_low",
                "standard_error",
                "upper_confidence_interval",
                "lower_confidence_interval",
                "dystopia_residual",
            ],
            strict=False,
        )
    )

    return schema_raw.validate(
        result,
        lazy=True,
    ).lazy()


@check_io(data=schema_raw, out=schema_parsed)
def fill_regions(data: pl.LazyFrame) -> pl.LazyFrame:
    """Fill regions for countries with a forward/backward fill"""
    return data.with_columns(
        pl.col("region")
        .forward_fill()
        .backward_fill()
        .over("country")
        # TODO: might only need one 'over()' here? as should be
        # able to represent a chained forward_fill/backward_fill
        # as a single expression, and .over operates on expressions?
        # good opportunity to learn more about polars expressions!
        .alias("region")
    )


@check_io(out=schema_parsed)
def read_data(folder: Path) -> pl.LazyFrame:
    return (
        pl.concat(
            [read_data_file(path) for path in folder.glob("*.csv")],
            how="diagonal_relaxed",
        )
        .sort(by=["country", "year"])
        .pipe(fill_regions)
        .pipe(schema_parsed.validate, lazy=True)
        .lazy()
        .select(col for col in schema_parsed.columns)
    )


@check_io(out=schema_stats)
def calc_stats(data: pl.LazyFrame) -> pl.LazyFrame:
    aggs = [
        pl.count("country").alias("num_countries"),
        # happiness
        pl.mean("happiness_score").alias("mean_happiness_score"),
        pl.min("happiness_score").alias("min_happiness_score"),
        pl.max("happiness_score").alias("max_happiness_score"),
        # gdp
        pl.mean("gdp_per_capita").alias("mean_gdp_per_capita"),
        pl.min("gdp_per_capita").alias("min_gdp_per_capita"),
        pl.max("gdp_per_capita").alias("max_gdp_per_capita"),
        # life expectancy
        pl.mean("healthy_life_expectancy").alias("mean_healthy_life_expectancy"),
        pl.min("healthy_life_expectancy").alias("min_healthy_life_expectancy"),
        pl.max("healthy_life_expectancy").alias("max_healthy_life_expectancy"),
        # family
        pl.mean("score_family_and_social_support").alias("mean_social_support"),
        pl.min("score_family_and_social_support").alias("min_social_support"),
        pl.max("score_family_and_social_support").alias("max_social_support"),
        # freedom
        pl.mean("score_freedom").alias("mean_freedom_score"),
        pl.min("score_freedom").alias("min_freedom_score"),
        pl.max("score_freedom").alias("max_freedom_score"),
        # corruption
        pl.mean("score_perception_of_corruption").alias(
            "mean_perception_of_corruption"
        ),
        pl.min("score_perception_of_corruption").alias("min_perception_of_corruption"),
        pl.max("score_perception_of_corruption").alias("max_perception_of_corruption"),
    ]

    yearly_stats = (
        data.group_by("year").agg(aggs).with_columns(happiness_quartile=pl.lit("All"))
    )

    yearly_quartile_stats = (
        data.with_columns(
            pl.col("happiness_score")
            .qcut(
                quantiles=[0.25, 0.5, 0.75],
                labels=["Q1", "Q2", "Q3", "Q4"],
            )
            .alias("happiness_quartile")
        )
        .group_by(["year", "happiness_quartile"])
        .agg(aggs)
        .sort(["year", "happiness_quartile"])
    )

    return pl.concat(
        [yearly_stats, yearly_quartile_stats], how="diagonal_relaxed"
    ).select(schema_stats.columns.keys())


def commentate(data: pl.DataFrame, stats: pl.DataFrame) -> None:
    import ollama
    import json

    COUNTRY = "Australia"
    YEAR = 2015
    year_data = data.filter(year=YEAR).with_columns(cs.float().round(2))

    global_json = json.dumps(
        stats.filter(year=YEAR, happiness_quartile="All")
        .with_columns(cs.float().round(2))
        .drop("year", "num_countries", "happiness_quartile")
        .to_dicts(),
        indent=2,
    )

    country_json = json.dumps(
        year_data.filter(country=COUNTRY).drop("region", "year").to_dicts(), indent=2
    )

    question = (
        f"You are an expert political analyst, specialising in happiness. Provide "
        f"a comprehensive summary on the {YEAR} World Happiness data for {COUNTRY} "
        f"using data for {COUNTRY} compared to global statistics\n"
        f"<{COUNTRY}>\n"
        f"{country_json}\n"
        f"</{COUNTRY}>\n\n"
        f"<Global Statistics>\n"
        f"{global_json}\n"
        f"</Global Statistics>\n"
    )

    print(f"asking ollama the following question:\n{question}")
    response = ollama.chat(
        model="llama3.2", messages=[{"role": "user", "content": question}]
    )

    reply = response["message"]["content"]

    print(f"reply:\n{reply}")

    return None


def demo() -> None:
    data = read_data(SOURCE_DIR)
    stats = calc_stats(data)

    commentate(data=data.collect(), stats=stats.collect())


if __name__ == "__main__":
    demo()
