from typing import List, Mapping, Union, Optional, Tuple
from warnings import warn

HTML_HEAD = \
    """
    <head>
        <title>Paraphrases overview</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <style type="text/css">
            .index-column {
                width: 30px;
                text-align: center;
                font-weight: bold;
            }
    
            .container {
                padding-top: 50px;
            }
    
            .example {
                transition: box-shadow .1s;
            }
            .example:hover {
                box-shadow: 0 0 3px rgba(33,33,33,.2);
            }
        </style>
        <script type="text/javascript">
            // Hides/shows system's metrics for an example. Expects row object as input.
            function toggleLocalMetrics(rowObject) {
                Array.from(rowObject.getElementsByClassName("local-metrics")).forEach(currMetrics => {
                    let currDisplayValue = getComputedStyle(currMetrics).display;
                    currMetrics.style.display = currDisplayValue !== "none" ? "none": "flex";
                })
            }
        </script>
    </head>
    """

_BOOTSTRAP_IMPORTS = \
    """
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script
    """


def multicolumn_visualization(column_values: List[List[str]],
                              column_names: List[str],
                              column_metric_data: Optional[List[Union[None, Mapping[str, List]]]] = None,
                              global_metric_data: Optional[List[Union[None, Mapping[str, int]]]] = None,
                              path: Optional[str] = None,
                              sort_by_system: Optional[Tuple[int, int]] = None,
                              sort_by_mean: Optional[int] = None):
    """ Displays an overview of paraphrases with a multi-column table-like visualization.

    Parameters
    ----------
    column_values:
        Output of different systems (or ground truth sequences). Each list should contain one sequence per example
    column_names:
        System names
    column_metric_data:
        Example-wise metrics
    global_metric_data:
        Dataset-wise metrics (such as averages, sizes, etc..). Should be provided for each column (can be None if there
        are no metrics)
    path:
        If provided, specifies where to save the HTML visualization
    sort_by_system:
        If provided, specifies which system's metric the rows should be sorted by in ascending order. First item in
        tuple is the system (column) index, second item is the metric index (in the order the local metrics are
        provided). Useful for exploring where a system does well and where it doesn't
    sort_by_mean:
        If provided, specifies which metric's mean value the rows should be sorted by in ascending order.
        Useful for exploring which examples are hard to solve well
    """
    assert len(column_values) == len(column_names)
    eff_col_metric_data = [None] * len(column_names) if column_metric_data is None else column_metric_data
    eff_global_metric_data = [None] * len(column_names) if global_metric_data is None else global_metric_data

    num_rows = len(column_values[0])
    for idx_col, sequences in enumerate(column_values):
        if len(sequences) != num_rows:
            raise ValueError(f"Mismatched number of sequences: "
                             f"System#{idx_col} has {len(sequences)}, expected {num_rows}")

    # Format meaning of columns (0th row)
    formatted_columns = []
    formatted_columns.append("<div class='index-column bg-light'></div>")
    for col_name in column_names:
        formatted_columns.append(f"<div class='col bg-light'><h5>{col_name}</h5></div>")
    formatted_columns = "<div class='row'>{}</div>".format("\n".join(formatted_columns))

    # Format systems' global metrics (1st row)
    formatted_global = []
    formatted_global.append("<div class='index-column'><strong>(G)</strong></div>")
    for curr_system_global in eff_global_metric_data:
        if curr_system_global is not None:
            inner_content = []
            for metric_name in sorted(curr_system_global.keys()):
                inner_content.append(f"<li>{metric_name}: {curr_system_global[metric_name]}</li>")

            inner_content = "<ul>{}</ul>".format("\n".join(inner_content))
        else:
            inner_content = ""

        formatted_global.append(f"<div class='col'>{inner_content}</div>")
    formatted_global = "<div class='row'>{}</div>".format("\n".join(formatted_global))

    # Figure out number of rows as data is given by columns
    for idx_col, system_local in enumerate(eff_col_metric_data):
        if system_local is None:
            continue

        for k, v in system_local.items():
            num_values_for_metric = len(v)
            if num_rows is not None and num_values_for_metric != num_rows:
                raise ValueError(
                    f"Not all metrics have same number of values: "
                    f"System#{idx_col}, metric '{k}' has {num_values_for_metric}, expected {num_rows})")

    _metrics = None
    # Convert column-wise data to row-wise
    eff_metric_names = []
    eff_metric_values = []
    for system_local in eff_col_metric_data:
        if system_local is None:
            eff_metric_names.append(None)
            eff_metric_values.append([None] * num_rows)
            continue

        _metrics = list(system_local)
        eff_metric_names.append(list(system_local))
        eff_metric_values.append(list(zip(*system_local.values())))
    eff_metric_values = list(zip(*eff_metric_values))

    if sort_by_system is not None:
        idx_system, idx_metric = sort_by_system
        sort_description = f"({column_names[idx_system]}, {_metrics[idx_metric]}), descending"
        mapped_values = list(map(lambda row: row[idx_system][idx_metric], eff_metric_values))
        sort_indices = [tup[0] for tup in sorted(enumerate(mapped_values), key=(lambda tup: -tup[1]))]
    elif sort_by_mean is not None:
        idx_metric = sort_by_mean
        sort_description = f"mean {_metrics[idx_metric]}, descending"
        mapped_values = []
        for idx_row, row_data in enumerate(eff_metric_values):
            numerator, denominator = 0, 0

            for col_data in row_data:
                if col_data is None:
                    continue

                numerator += col_data[idx_metric]
                denominator += 1

            if denominator == 0:
                warn(f"Found no valid metric values for metric '{_metrics[idx_metric]}' for example#{idx_row}")

            mapped_values.append(numerator / max(1, denominator))

        print(list(enumerate(mapped_values)))
        sort_indices = [tup[0] for tup in sorted(enumerate(mapped_values), key=(lambda tup: -tup[1]))]
    else:
        sort_indices = list(range(num_rows))
        sort_description = "index, ascending"

    formatted_rows = []
    for idx_row in sort_indices:
        curr_row_data = eff_metric_values[idx_row]
        curr_fmt = []
        curr_fmt.append(f"<div class='index-column'>{idx_row}.</div>")
        for idx_col in range(len(curr_row_data)):
            num_metrics = 0 if eff_metric_names[idx_col] is None else len(eff_metric_names[idx_col])

            fmt_metrics = ""
            fmt_text = f"<div class='row'><div class='col'>{column_values[idx_col][idx_row]}</div></div>"
            if num_metrics > 0:
                fmt_metrics = []
                for metric_name in eff_metric_names[idx_col]:
                    fmt_metrics.append(f"<div class='col'><strong>{metric_name}</strong></div>")
                fmt_metrics.append(f"<div class='w-100'></div>")
                for metric_value in curr_row_data[idx_col]:
                    fmt_metrics.append(f"<div class='col''>{round(metric_value, 4) if isinstance(metric_value, float) else metric_value}</div>")

                fmt_metrics = "<div class='row local-metrics'>{}</div>".format("\n".join(fmt_metrics))

            curr_fmt.append("<div class='col'>{}{}</div>".format(fmt_text, "".join(fmt_metrics)))

        formatted_rows.append(
            "<div class='row example'>{}</div>".format("\n".join(curr_fmt)))
    formatted_rows = "\n".join(formatted_rows)

    result_html = \
        f"""
        <!DOCTYPE html>
        <html>
            {HTML_HEAD}
            <body>
                <div class='container'>
                <div class='row'>
                    <h2>
                    Comparison of paraphrases
                    <small class='text-muted'>({num_rows} examples)</small>
                    </h2>
                </div>
                <div class='row'>
                    <small class='text-muted'>Sorted by: {sort_description}</small>
                </div>
                {formatted_columns}
                {formatted_global}
                {formatted_rows}
                </div>
            </body>
            {_BOOTSTRAP_IMPORTS}
        </html>
        """

    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            print(result_html, file=f)

    return result_html


if __name__ == "__main__":
    INPUT_SENTENCES = ["A dog swims in a body of water.",
                       "A group of tourist waiting for a train outside."]
    TARGET_SENTENCES = ["Closeup image of a dog swimming.",
                        "Tourists waiting at a train stop."]
    SYSTEM1 = ["A dog is swimming in a lake.",
               "There are tourists waiting."]
    SYSTEM2 = ["A dog is swimming.",
               "There are tourists are tourists are tourists are tourists are tourists are tourists"]

    multicolumn_visualization(column_values=[INPUT_SENTENCES, TARGET_SENTENCES, SYSTEM1, SYSTEM2],
                              column_names=["Input", "Ground truth", "System 1", "System 2"],
                              column_metric_data=[None, None, {"BERT": [0.25, 0.7]}, {"BERT": [0.5, 0.6], "BLEU": [0.3, 0.5]}],
                              global_metric_data=[None, None, {"BERT": 13.3}, {"BERT": 123, "BLEU": 1.2}],
                              path="visualization.html")
