
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
        names of models, i.e. ['parent1', ...]
    caption: str
        table caption describing the table
    label: str
        reference for the table
    path: str
        filename for saving the table, e.g. 'table_traces.txt'
    """
    idt = lambda n: '  ' * n # indent by multiples of 2 spaces
    alignment = '{' + ('l ' + 'r ' * (len(col_names) - 1)).strip() + '}'
    headers = '~ ' + ' & '.join(col_names) + r' \\'

    content = ''
    for e_id, e_dict in enumerate(dicts):
        values = ' & '.join(map(str, [ e_dict[col] for col in col_names ]))
        row = col_names[e_id] + ' & ' + values + r' \\'
        content += idt(3) + row + '\n'
    content = content.strip() # remove first indent

    table = fr"""\begin{{table}}[!ht]
      \centering
        \begin{{tabular}}{alignment}
          \toprule
          {headers}
        \midrule
          {content}
        \bottomrule
      \end{{tabular}}
      \caption{{{caption}}}
      \label{{{label}}}
    \end{{table}}
    """

    with open(path, 'w') as f:
        f.write(table)


