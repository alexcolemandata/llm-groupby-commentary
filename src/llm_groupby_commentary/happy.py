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
        "mean_gdp_per_capita": pa.Column(float),
        "mean_healthy_life_expectancy": pa.Column(float),
        "mean_social_support": pa.Column(float),
        "mean_freedom_score": pa.Column(float),
        "min_happiness_score": pa.Column(float),
        "max_happiness_score": pa.Column(float),
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
        pl.mean("happiness_score").alias("mean_happiness_score"),
        pl.mean("gdp_per_capita").alias("mean_gdp_per_capita"),
        pl.mean("healthy_life_expectancy").alias("mean_healthy_life_expectancy"),
        pl.mean("score_family_and_social_support").alias("mean_social_support"),
        pl.mean("score_freedom").alias("mean_freedom_score"),
        pl.min("happiness_score").alias("min_happiness_score"),
        pl.max("happiness_score").alias("max_happiness_score"),
        pl.count("country").alias("num_countries"),  # Count countries per year
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


if __name__ == "__main__":
    data = read_data(SOURCE_DIR)
    stats = calc_stats(data)

    print(f"data:\n{data.collect()}\n\nstats:\n{stats.collect()}\n\n")
