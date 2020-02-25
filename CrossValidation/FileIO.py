import os


def available_file_path(export_file_path):
    exists = os.path.isfile(export_file_path)
    if exists:
        path = export_file_path.split('.csv')[0]  # chop off ending
        i = 0
        while exists:
            i = i + 1
            export_file_path = path + '_' + str(i) + '.csv'
            exists = os.path.isfile(export_file_path)
    os.makedirs(os.path.dirname(export_file_path), exist_ok=True)
    return export_file_path


def record_rms_run(dst, moth, rms_values, **kwargs):
    summary = summarize(moth, **kwargs)
    f = open(dst, 'a')
    f.write(summary)
    rms_values.to_csv(f, index=False)
    f.close()


def summarize(moth, **kwargs):
    summary = "#moth=" + moth + ","
    for key, value in kwargs.items():
        summary = summary + str(key) + '=' + str(value) + ","
    return summary[0:-1] + '\n'