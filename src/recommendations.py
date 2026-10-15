def generate_recommendation(risk_probability):


    if not 0 <= risk_probability <= 1:
        raise ValueError(
            "Risk probability must be between 0 and 1."
        )

    if risk_probability >= 0.75:
        return {
            "risk_level": "HIGH",
            "recommendation": (
                "Prioritize this order for manual review. "
                "Check carrier capacity, inventory readiness, "
                "and available alternative shipping options."
            ),
        }

    if risk_probability >= 0.40:
        return {
            "risk_level": "MEDIUM",
            "recommendation": (
                "Monitor shipment preparation and carrier progress. "
                "Review possible inventory or fulfillment bottlenecks."
            ),
        }

    return {
        "risk_level": "LOW",
        "recommendation": (
            "Continue standard fulfillment monitoring and "
            "reassess the order if new risk signals appear."
        ),
    }


if __name__ == "__main__":
    example_probability = 0.82

    result = generate_recommendation(example_probability)

    print("Predicted risk probability:", example_probability)
    print("Risk level:", result["risk_level"])
    print("Recommended action:", result["recommendation"])