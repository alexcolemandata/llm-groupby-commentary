# llm-grouby-commentary

A learning project for generating automatic commentary for polars groupby objects
using ollama

## Demo - Verbose and Spurious Analysis for Australia 2015 Happiness

```
> llm-groupby-commentary
asking ollama the following question:

You are an expert political analyst, specialising in happiness. Provide a
comprehensive summary on the 2015 World Happiness data for Australia using data
for Australia compared to global statistics

<Australia>
[
  {
    "country": "Australia",
    "happiness_rank": 10,
    "happiness_score": 7.28,
    "gdp_per_capita": 1.33,
    "healthy_life_expectancy": 0.93,
    "score_family_and_social_support": 1.31,
    "score_freedom": 0.65,
    "score_perception_of_corruption": 0.36,
    "score_generosity": 0.44
  }
]
</Australia>

<Global Statistics>
[
  {
    "mean_happiness_score": 5.38,
    "min_happiness_score": 2.84,
    "max_happiness_score": 7.59,
    "mean_gdp_per_capita": 0.85,
    "min_gdp_per_capita": 0.0,
    "max_gdp_per_capita": 1.69,
    "mean_healthy_life_expectancy": 0.63,
    "min_healthy_life_expectancy": 0.0,
    "max_healthy_life_expectancy": 1.03,
    "mean_social_support": 0.99,
    "min_social_support": 0.0,
    "max_social_support": 1.4,
    "mean_freedom_score": 0.43,
    "min_freedom_score": 0.0,
    "max_freedom_score": 0.67,
    "mean_perception_of_corruption": 0.14,
    "min_perception_of_corruption": 0.0,
    "max_perception_of_corruption": 0.55
  }
]
</Global Statistics>

reply:
Based on the 2015 World Happiness data, Australia ranks 10th in terms of overall happiness score with a score of 7.28 out of 10. Here's a comprehensive summary comparing Australia to global statistics:

**Happiness Score:**

* Australia (rank 10): 7.28
* Global Average: 5.38

Australia scores significantly higher than the global average, indicating that Australians tend to report higher levels of happiness compared to people worldwide.

**Economic Factors:**

* GDP per capita:
 + Australia: $1.33 (ranked 12th globally)
 + Global Average: $0.85
 While Australia's GDP per capita is lower than the global average, its ranking in this category is still relatively high.

**Health and Well-being:**

* Healthy life expectancy:
 + Australia: 0.93
 + Global Average: 0.63
 Australians have a higher healthy life expectancy compared to the global average, indicating better overall health and well-being.
* Social support:
 + Australia: 1.31 (ranked 5th globally)
 + Global Average: 0.99
 Australians score above the global average in terms of social support, suggesting strong family and social connections.

**Freedom and Governance:**

* Freedom score:
 + Australia: 0.65 (ranked 26th globally)
 + Global Average: 0.43
 While Australia's freedom score is lower than the global average, its ranking is still relatively high compared to other countries.
* Perception of corruption:
 + Australia: 0.36
 + Global Average: 0.14
 Australians have a higher perception of corruption compared to the global average, indicating concerns about government transparency and accountability.

**Social Values:**

* Generosity score:
 + Australia: 0.44 (ranked 24th globally)
 + Global Average: Not available
 Australia's generosity score is lower than the global average in this category, suggesting a need for more charitable giving and community engagement.

In conclusion, while Australia ranks high in terms of overall happiness, there are areas where it lags behind the global average. Improving its GDP per capita, perception of corruption, and social values could potentially boost Australia's ranking further.
```
