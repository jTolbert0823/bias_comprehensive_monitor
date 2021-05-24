import modelop.monitors.bias as bias
import modelop.schema.infer as infer
import modelop.utils as utils

logger = utils.configure_logger()

MONITORING_PARAMETERS = {}

# modelop.init
def init(job_json):
    """A function to extract input schema from job JSON.
    Args:
        job_json (str): job JSON in a string format.
    """

    # Extract input schema from job JSON
    input_schema_definition = infer.extract_input_schema(job_json)

    logger.info("Input schema definition: %s", input_schema_definition)

    # Get monitoring parameters from schema
    global MONITORING_PARAMETERS
    MONITORING_PARAMETERS = infer.set_monitoring_parameters(
        schema_json=input_schema_definition, check_schema=True
    )

    logger.info("label_column: %s", MONITORING_PARAMETERS["label_column"])
    logger.info("score_column: %s", MONITORING_PARAMETERS["score_column"])


# modelop.metrics
def metrics(dataframe):

    logger.info(
        "protected_classes: %s", str(MONITORING_PARAMETERS["protected_classes"])
    )

    result = {"bias": []}
    for feature in MONITORING_PARAMETERS["protected_classes"]:
        # Initialize BiasMonitor
        bias_monitor = bias.BiasMonitor(
            dataframe=dataframe,
            score_column=MONITORING_PARAMETERS["score_column"],
            label_column=MONITORING_PARAMETERS["label_column"],
            protected_class=feature,
            reference_group=None,
        )

        # Compute aequitas_bias (disparity) metrics
        bias_metrics = bias_monitor.compute_bias_metrics(
            pre_defined_test="aequitas_bias", thresholds={"min": 0.8, "max": 1.2}
        )

        result["bias"].append(bias_metrics)

        # Compute aequitas_group (Group) metrics
        group_metrics = bias_monitor.compute_group_metrics(
            pre_defined_test="aequitas_group",
        )

        result["bias"].append(group_metrics)

    yield result
