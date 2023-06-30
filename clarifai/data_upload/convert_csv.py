import csv
import os
import sys
from itertools import chain

import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:

  def __init__(self, filename, multi_val, separator):
    self.filename = filename
    self.multi_val = multi_val
    self.separator = separator
    self.df = pd.read_csv(filename)

  def process_data(self):
    text_col = self.get_column("Enter the number of the column that contains the text: ")
    label_col = self.get_column(
        "Enter the number of the column that contains the labels: ", exclude=text_col)

    self.split_values(label_col)
    self.df[label_col] = self.df[label_col].apply(lambda x: [x] if isinstance(x, str) else x)

    # Use chain.from_iterable to expand the multi-values if applicable
    unique_labels = list(set(chain.from_iterable(self.df[label_col].values)))

    print(
        "\nThe following unqiue labels have been found in the '{}' column and will be used in the dataset:".
        format(label_col))
    for i, label in enumerate(unique_labels, start=1):
      print('{}) {}'.format(i, label))

    self.convert_for_classification(text_col, label_col, unique_labels)

  def get_column(self, prompt, exclude=None):
    available_columns = self.df.columns.drop(exclude) if exclude else self.df.columns
    if len(available_columns) == 1:
      print(f'\nThe column named \'{available_columns[0]}\' will be used as the labels column.')
      return available_columns[0]
    else:
      for i, col in enumerate(available_columns):
        print(f'{i+1}) {col}')
      col_index = int(input(prompt)) - 1
      return available_columns[col_index]

  def split_values(self, label_col):
    if self.multi_val.lower() == 'y':
      self.df[label_col] = self.df[label_col].apply(
          lambda x: str(x).split(self.separator) if not isinstance(x, float) else x)

  def convert_for_classification(self, text_col, label_col, unique_labels):
    # Binary classification
    if len(unique_labels) == 2:
      print("Converting the CSV to be used with binary classification")
      self.df['input.data.text.raw'] = self.df[text_col]
      self.df['input.data.concepts[0].id'] = label_col
      self.df['input.data.concepts[0].value'] = self.df[label_col].apply(
          lambda x: 1 if unique_labels[0] in x else 0)
      self.df = self.df[[
          'input.data.text.raw', 'input.data.concepts[0].id', 'input.data.concepts[0].value'
      ]]

    # Multi-class classification
    else:
      print("Converting the CSV to be used with multi-class classification")
      self.df['input.data.text.raw'] = self.df[text_col].apply(
          lambda x: x[0] if isinstance(x, list) else x)
      for i in range(len(unique_labels)):
        self.df[f'input.data.concepts[{i}].id'] = self.df[label_col].apply(
            lambda x: unique_labels[i] if unique_labels[i] in x else '')
        self.df[f'input.data.concepts[{i}].value'] = self.df[label_col].apply(
            lambda x: 1 if unique_labels[i] in x else '')

      self.df = self.df[['input.data.text.raw'] +
                        [f'input.data.concepts[{i}].id' for i in range(len(unique_labels))] +
                        [f'input.data.concepts[{i}].value' for i in range(len(unique_labels))]]

      # Reorder the columns
      cols = self.df.columns.tolist()
      new_cols = cols[:1]  # The first column 'input.data.text.raw'
      pairs = [[cols[i], cols[i + len(unique_labels)]] for i in range(1, len(unique_labels) + 1)]
      for pair in pairs:
        new_cols.extend(pair)
      self.df = self.df[new_cols]

      # Remove special characters from column names
      self.df.columns = self.df.columns.str.replace("^[\[]|[\]]$", "", regex=True)


class DatasetSplitter:

  def __init__(self, df, split_dataset, shuffle_dataset, seed=555):
    self.df = df
    self.split_dataset = split_dataset
    self.shuffle_dataset = shuffle_dataset
    self.seed = seed if seed != '' else 555

  def split_and_save(self, filename_base):
    if self.split_dataset.lower() == 'y':
      split_type = self.get_split_type()

      if split_type == 1:
        train_pct = self.get_percentage(
            'What percentage of the dataset should be used for training? Enter a number between 1 and 99: ',
            99)
        test_pct = 100 - train_pct
        print(f'Data will be split {train_pct}% train, {test_pct}% test')  # Added print statement
      elif split_type == 2:
        train_pct = self.get_percentage(
            'What percentage of the dataset should be used for training? Enter a number between 1 and 98: ',
            98)
        max_val_pct = 99 - train_pct  # Max percentage for validation is now reduced by 1
        val_pct = self.get_percentage(
            f'What percentage of the dataset should be used for validation? Enter a number between 1 and {max_val_pct}: ',
            max_val_pct)
        test_pct = 100 - train_pct - val_pct
        print(f'Data will be split {train_pct}% train, {val_pct}% validation, {test_pct}% test'
             )  # Added print statement

      train_df, test_df = train_test_split(
          self.df,
          test_size=test_pct / 100,
          random_state=self.seed,
          shuffle=self.shuffle_dataset.lower() == 'y')
      train_df.to_csv(filename_base + '-train.csv', index=False, quoting=csv.QUOTE_MINIMAL)
      test_df.to_csv(filename_base + '-test.csv', index=False, quoting=csv.QUOTE_MINIMAL)

      if split_type == 2:
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_pct / 100,
            random_state=self.seed,
            shuffle=self.shuffle_dataset.lower() == 'y')
        val_df.to_csv(filename_base + '-validation.csv', index=False, quoting=csv.QUOTE_MINIMAL)
    else:
      self.df.to_csv(filename_base + '.csv', index=False, quoting=csv.QUOTE_MINIMAL)

  def get_split_type(self):
    split_type = int(
        input(
            'How would you like to split the dataset?\n1) Train and test datasets\n2) Train, validate, and test datasets\n'
        ))
    while split_type not in [1, 2]:
      split_type = int(
          input(
              'Invalid option. Enter 1 for "Train and test" or 2 for "Train, validate, and test": '
          ))
    return split_type

  def get_percentage(self, prompt, max_pct):
    pct = int(input(prompt))
    while not 1 <= pct <= max_pct:
      pct = int(input(f'Invalid input. Please enter a number between 1 and {max_pct}: '))
    return pct


def main():
  filename = sys.argv[1]
  multi_val = input('Do any columns have multiple values? (y/[n]) ')
  separator = input('Enter the separator: ') if multi_val.lower() == 'y' else None

  preprocessor = DataPreprocessor(filename, multi_val, separator)
  preprocessor.process_data()

  split_dataset = input('Would you like to split this dataset? (y/[n]) ')
  shuffle_dataset = 'n'
  seed = '555'

  if split_dataset.lower() == 'y':
    shuffle_dataset = input('Would you like to shuffle the dataset before splitting? (y/[n]) ')
    if shuffle_dataset.lower() == 'y':
      seed = input('Enter a seed integer or hit enter to use the default [555]: ')

  splitter = DatasetSplitter(preprocessor.df, split_dataset, shuffle_dataset, seed)
  splitter.split_and_save(os.path.splitext(filename)[0] + '-clarifai')
  print("Done!")


if __name__ == "__main__":
  main()
