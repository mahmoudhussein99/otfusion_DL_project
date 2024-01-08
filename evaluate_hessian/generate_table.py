
def output_table(row_names, col_names, dicts, caption, label, path):
    """
    can be used for:
    - accuracy / loss table
    - trace table
    - top K eigenvalues table

    dicts: list[dict]
        list of dicts that have keys as col_names with one string value per key,
    row_names: list[str]
        normally names of experiments
    col_names: list[str]
        names of models, i.e. ['parent 1', ...] for pretty table headers
    caption: str
        table caption describing the table
    label: str
        reference for the table
    path: str
        filename for saving the table, e.g. 'table_traces.txt'
    """
    idt = lambda n: '  ' * n # indent by multiples of 2 spaces
    alignment = '{' + ('l ' + 'c ' * len(col_names)).strip() + '}'
    headers = '~ & ' + ' & '.join(col_names) + r' \\'
    dict_cols = dicts[0].keys()

    content = ''
    for e_id, e_dict in enumerate(dicts):
        values = ' & '.join(map(str, [ e_dict[col] for col in dict_cols ]))
        row = row_names[e_id] + ' & ' + values + r' \\'
        content += idt(3) + row + '\n'
    content = content.strip() # remove first indent

    table = fr"""
\begin{{table}}[!ht]
  \centering\small\resizebox{{0.5\textwidth}}{{!}}{{
  \begin{{tabular}}{alignment}
  \toprule
    {headers}
    \midrule
    {content}
    \bottomrule
  \end{{tabular}}
}}
\caption{{{caption}}}
\label{{{label}}}
\end{{table}}
"""

    with open(path, 'w') as f:
        f.write(table)


