"""World happiness data"""

from pathlib import Path

import polars as pl
import re
import pandera.polars as pa

from polars.datatypes import Int32
import polars.selectors as cs

from llm_groupby_commentary import DATA_DIR

SOURCE_DIR = DATA_DIR / "world_happiness"

raw_schema = pa.DataFrameSchema(
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
    strict="filter",
)

parsed_schema = raw_schema.update_column("region", required=True)


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

    return raw_schema.validate(
        result,
        lazy=True,
    ).lazy()


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


def read_data(folder: Path) -> pl.LazyFrame:
    return (
        pl.concat(
            [read_data_file(path) for path in folder.glob("*.csv")],
            how="diagonal_relaxed",
        )
        .sort(by=["country", "year"])
        .pipe(fill_regions)
        .pipe(parsed_schema.validate, lazy=True)
        .lazy()
        .select(col for col in parsed_schema.columns)
    )


if __name__ == "__main__":
    print(read_data(SOURCE_DIR).collect())
