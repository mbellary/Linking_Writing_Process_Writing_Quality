 Credit:
# This work is based on the top  Kernel from the competition : https://www.kaggle.com/code/awqatak/silver-bullet-single-model-165-features
# Only minor modifications to few functions



import polars as pl
import pandas as pd
import re
from scipy.stats import skew, kurtosis

num_cols = ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']
activities = ['Input', 'Remove', 'Nonproduction', 'Replace', 'Paste']
events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
text_changes = ['q', ' ', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
target_cols = ['activity', 'text_change', 'down_event', 'up_event']



def count_values(df, colnames, colvalues):
    index = pl.col('id')
    values = pl.col('count')
    agg_func = lambda col : col.first()
    for colname, unique_values in zip(colnames, colvalues):
        on = pl.col(colname)
        df_temp = (df.filter(pl.col(colname).is_in(unique_values))
                            .group_by(index).agg(pl.col(colname).value_counts(sort=True))
                            .explode(colname)
                            .unnest(colname)
                            .group_by(index).agg(agg_func(values.filter(on == value)).alias(f'{colname}_{value}') 
                                                 for value in unique_values).fill_null(0))
        df = df.join(df_temp, on="id", how="left")
    return df

def input_text_stats(df):
    df_temp = (df.filter((~pl.col('text_change').str.contains('=>')) & (pl.col('text_change') != 'NoChange'))
                  .group_by('id').agg(pl.col('text_change').str.concat('').str.extract_all(r'q+'))
                  .with_columns(
                        text_change_len = pl.col('text_change').list.eval(pl.element().str.len_chars()))
                  .with_columns(
                        input_word_count = pl.col('text_change_len').list.len(),
                        input_word_length_mean = pl.col('text_change_len').list.mean(),
                        input_word_length_max = pl.col('text_change_len').list.max(),
                        input_word_length_std = pl.col('text_change_len').list.std(),
                        input_word_length_median = pl.col('text_change_len').list.median(),
                        input_word_length_skew = pl.col('text_change_len').map_elements(skew, return_dtype=pl.Float64)))
    df = df.join(df_temp, on="id", how="left")
    return df

def numerical_col_features(df, num_cols):
    df_temp = (df.group_by('id').agg(
                    pl.sum('action_time').name.suffix('_sum'), 
                    pl.mean(num_cols).name.suffix('_mean'), 
                    pl.std(num_cols).name.suffix('_std'),
                    pl.median(num_cols).name.suffix('_median'), 
                    pl.min(num_cols).name.suffix('_min'), 
                    pl.max(num_cols).name.suffix('_max'),
                    pl.quantile(num_cols, 0.5).name.suffix('_quantile')))
    df = df.join(df_temp, on="id", how="left")
    return df

def categorical_col_features(df):
    df_temp  = df.group_by("id").agg(pl.n_unique(['activity', 'down_event', 'up_event', 'text_change']))
    df = df.join(df_temp, on='id', how='left')
    return df

def idle_time_features(df):
    df_temp = (train_df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
                    .with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
                    .filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
                    .group_by("id").agg(
                       inter_key_largest_lantency = pl.max('time_diff'),
                       inter_key_median_lantency = pl.median('time_diff'),
                       mean_pause_time = pl.mean('time_diff'),
                       std_pause_time = pl.std('time_diff'),
                       total_pause_time = pl.sum('time_diff'),
                       pauses_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 0.5) & (pl.col('time_diff') < 1)).count(),
                       pauses_1_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1) & (pl.col('time_diff') < 1.5)).count(),
                       pauses_1_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1.5) & (pl.col('time_diff') < 2)).count(),
                       pauses_2_sec = pl.col('time_diff').filter((pl.col('time_diff') > 2) & (pl.col('time_diff') < 3)).count(),
                       pauses_3_sec = pl.col('time_diff').filter(pl.col('time_diff') > 3).count()))
    df = df.join(df_temp, on='id', how='left')
    return df

def p_bursts_feature(df):
    df_temp = (df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
                    .with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
                    .filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
                    .with_columns(pl.col('time_diff')<2)
                    .with_columns(pl.when(pl.col("time_diff") & pl.col("time_diff").is_last_distinct()).then(pl.len()).over(pl.col("time_diff").rle_id()).alias('P-bursts'))
                    .drop_nulls()
                    .group_by("id").agg(
                        pl.mean('P-bursts').name.suffix('_mean'), 
                        pl.std('P-bursts').name.suffix('_std'), 
                        pl.count('P-bursts').name.suffix('_count'),
                        pl.median('P-bursts').name.suffix('_median'), 
                        pl.max('P-bursts').name.suffix('_max'),
                        pl.first('P-bursts').name.suffix('_first'), 
                        pl.last('P-bursts').name.suffix('_last')))
    df = df.join(df_temp, on='id', how='left')
    return df

def r_bursts_feature(df):
    df_temp = (df.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
                 .with_columns(pl.col('activity').is_in(['Remove/Cut']))
                 .with_columns(pl.when(pl.col("activity") & pl.col("activity").is_last_distinct()).then(pl.len()).over(pl.col("activity").rle_id()).alias('R-bursts'))
                 .drop_nulls()
                 .group_by("id").agg(
                     pl.mean('R-bursts').name.suffix('_mean'), 
                     pl.std('R-bursts').name.suffix('_std'), 
                     pl.median('R-bursts').name.suffix('_median'), 
                     pl.max('R-bursts').name.suffix('_max'),
                     pl.first('R-bursts').name.suffix('_first'), 
                     pl.last('R-bursts').name.suffix('_last')))
    df = df.join(df_temp, on='id', how='left')
    return df
    
# Essay reconstruction using Pandas 
def reconstruct_essay(currTextInput):
    essayText = ""
    for Input in currTextInput.values:
        if Input[0] == 'Replace':
            replaceTxt = Input[2].split(' => ')
            essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
            continue
        if Input[0] == 'Paste':
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
            continue
        if Input[0] == 'Remove/Cut':
            essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
            continue
        if "M" in Input[0]:
            croppedTxt = Input[0][10:]
            splitTxt = croppedTxt.split(' To ')
            valueArr = [item.split(', ') for item in splitTxt]
            moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
            if moveData[0] != moveData[2]:
                if moveData[0] < moveData[2]:
                    essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                else:
                    essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
            continue
        essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
    return essayText


def get_essay_df(df):
    df       = df[df.activity != 'Nonproduction']
    temp     = df.groupby('id').apply(lambda x: reconstruct_essay(x[['activity', 'cursor_position', 'text_change']]))
    essay_df = pd.DataFrame({'id': df['id'].unique().tolist()})
    essay_df = essay_df.merge(temp.rename('essay'), on='id')
    return essay_df


def split_essays(essay, feat_type):
    if feat_type == 'word':
        res = re.split(' |\\n|\\.|\\?|\\!', essay)
    elif feat_type == 'sent':
        res = re.split('\\.|\\?|\\!', essay)
    elif feat_type == 'para':
        res = essay.split('\n')
    return res

def get_token_len(feat_types):
    for feat_type in feat_types:
        if feat_type != 'sent_word':
            yield pl.col(feat_type).list.eval(pl.element().str.len_chars()).alias(f"{feat_type}_len")
        else:
            yield pl.col(feat_type).list.eval(pl.element().list.eval(pl.element().len())).alias(f"{feat_type}_len")

def get_essay_stats(df, feat_cols):
    for col in feat_cols:
        if col != 'sent_word_len':
            df_feats = df.select(
                            pl.col('id'),
                            pl.col(col).list.len().alias(f"{col}_count"),
                            pl.col(col).list.mean().alias(f"{col}_mean"),
                            pl.col(col).list.min().alias(f"{col}_min"),
                            pl.col(col).list.max().alias(f"{col}_max"),
                            pl.col(col).list.first().alias(f"{col}_first"),
                            pl.col(col).list.last().alias(f"{col}_last"),
                            pl.col(col).explode().explode().quantile(0.25).alias(f"{col}_q1"),
                            pl.col(col).list.median().alias(f"{col}_median"),
                            pl.col(col).explode().explode().quantile(0.75).alias(f"{col}_q3"), # list[list[str]]
                            pl.col(col).list.sum().alias(f"{col}_sum"))
        else:
            df_feats = df.select(
                            pl.col('id'),
                            pl.col(col).list.eval(pl.element().len()).alias(f"{col}_count"),
                            pl.col(col).list.eval(pl.element().mean()).alias(f"{col}_mean"),
                            pl.col(col).list.eval(pl.element().min()).alias(f"{col}_min"),
                            pl.col(col).list.eval(pl.element().max()).alias(f"{col}_max"),
                            pl.col(col).list.eval(pl.element().first()).alias(f"{col}_first"),
                            pl.col(col).list.eval(pl.element().last()).alias(f"{col}_last"),
                            pl.col(col).explode().explode().quantile(0.25).alias(f"{col}_q1"),
                            pl.col(col).list.eval(pl.element().median()).alias(f"{col}_median"),
                            pl.col(col).explode().explode().quantile(0.75).alias(f"{col}_q3"), # list[list[str]]
                            pl.col(col).list.eval(pl.element().sum()).list.sum().alias(f"{col}_sum"))
        df = df.join(df_feats, on='id', how='left')
    return df

def reshape_sent_word(df):
    df_temp = df.select(
                pl.col('id'), 
                pl.col("sent_word_len")
            ).explode("sent_word_len").group_by('id').agg(pl.col("sent_word_len").flatten().alias("sent_word_len2"))
    df = df.join(df_temp, on='id', how='left')
    df = df.drop("sent_word_len").rename({"sent_word_len2" : "sent_word_len"})
    return df

def key_stroke_feats(df):
    df_feats = (df.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
                 .group_by('id').agg(
                     pl.col('activity').len().alias('key_pressed_1'),
                     pl.col('event_id').len().alias('key_pressed_2'),
                     pl.col('down_time').min().alias('min_down_time'),
                     pl.col('up_time').max().alias('max_up_time')))
    df = df.join(df_feats, on='id', how='left')
    df_temp = ( df.with_columns(pl.col('essay').str.len_chars().alias('product_len'))
                .with_columns( (pl.col("product_len") / pl.col('key_pressed_1')).alias('product_to_keys'),
                                (pl.col("key_pressed_2") / ((pl.col('max_up_time') - (pl.col(min_down_time))) / 1000 )).alias('keys_per_second')))


 def main(dir_path):
 	train_df = pl.scan_csv(dir_path + 'dataset/linking-writing-processes-to-writing-quality/train_logs.csv')
 	df_pandas = pd.read_csv(dir_path + 'dataset/linking-writing-processes-to-writing-quality/train_logs.csv')
	train_essays = get_essay_df(df_pandas)
	train_essays = pl.from_pandas(train_essays).lazy()
	train_df = (train_df
			     .pipe(count_values, colnames=target_cols, colvalues=[activities, text_changes, events, events])
			     .pipe(input_text_stats)
			     .pipe(numerical_col_features, num_cols=num_cols)
			     .pipe(categorical_col_features)
			     .pipe(idle_time_features)
			     .pipe(p_bursts_feature)
			     .pipe(r_bursts_feature) )
	train_essays = (train_essays.with_columns(
                                pl.col('essay').map_elements(lambda essay: split_essays(essay, 'word'), return_dtype=pl.List(str)).alias("word"),
                                pl.col('essay').map_elements(lambda essay: split_essays(essay, 'sent'), return_dtype=pl.List(str)).alias("sent"),
                                pl.col('essay').map_elements(lambda essay: split_essays(essay, 'para'), return_dtype=pl.List(str)).alias("para"))
                           .with_columns(pl.col('sent').list.eval(pl.element().str.replace('\n', '').str.strip_chars()))
                           .with_columns(pl.col('sent').list.eval(pl.element().str.split(' ')).alias('sent_word'))
                           .with_columns(get_token_len(['word', 'sent', 'sent_word', 'para']))
                           .pipe(reshape_sent_word)
                           .pipe(get_essay_stats, feat_cols=['word_len', 'sent_len', 'sent_word_len', 'para_len'])
                           .join(train_df, on="id")
                           .pipe(key_stroke_feats))

if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Preprocess training data')
parser.add_argument('--dir_path', metavar='path', required=True, help='the path to data files')
args = parser.parse_args()
main(args.dir_path)