# Functions for animated bar charts

def show_wordcloud(data, stopwords, title = None, ax=None):
    """Show wordcloud.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with data to be plotted.
    stopwords : list
        Words to be removed from the plot.
    title : str, optional
        Title of the plot.
    """
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=1000, 
        max_font_size=40, 
        scale=3,
        random_state=1
    ).generate(str(data))

    ax.axis('off')
    if title: 
        ax.set_title(title, fontsize=20)

    ax.imshow(wordcloud)



def filter_data(df, subjects=[], institutions=[], years=[0, 2022]):
    """Filter data for animated bar charts.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with data.
    subjects : str, optional
        Subjects to be included.
    years : list, optional
        Years to be included.

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe.
    """
    
    # Filter by subject
    if len(subjects) > 0:
        df = df[df['Subject Discipline'].isin(subjects)]

    # Filter by institution
    if len(institutions) > 0:
        df = df[df['Institution'].isin(institutions)]

    # Filter by year
    df = df.loc[df['Date'] != ' ']
    df['Year'] = df['Date'].astype(int)
    df = df.loc[(df['Year'] != ' ') & (df['Year'].astype(int) >= years[0]) & (df['Year'].astype(int) <= years[1])]

    return df


def preprocess_data2(df, subjects=[], institutions=[], stop_words=[], word_column='Title', all_lower=True):
    """Preprocess data for animated bar charts.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with data to be plotted.
    subjects : str, optional
        Subjects to be plotted.
    stopwords : list, optional
        Words to be removed from the plot.
    word_coloumn : str, optional
        Column name of the dataframe that contains the words to be plotted.
    all_lower : bool, optional
        If True, all words are converted to lowercase.

    Returns
    -------
    pandas.DataFrame
        Preprocessed dataframe.
    """
    df_wc = filter_data(df, subjects, institutions)

    # Lowercase words
    if all_lower: 
        df_wc[word_column] = df_wc[word_column].str.lower()

    # Remove punctuation
    import string 
    df_wc['words'] = df_wc[word_column]
    #df_wc['words'] = str(df_wc['words'])
    #

    df_wc['words'] = df_wc['words'].str.replace('[{}]'.format(string.punctuation), '', regex=True)  
    
    #df_wc['words'] = df_wc['words'].str.partition('\n')[0] # remove everything after the first line break
    
    #print(list(df_wc.loc[df_wc['words'].str.contains('2'), ['words']])[1])
    #print(df_wc.head())

    # Number of theses per year
    thesis_per_year = df_wc['Year'].value_counts()
    thesis_per_year=thesis_per_year.to_frame('Count').reset_index()
    thesis_per_year.columns = ['Year', 'total_publications']    
    
    # Split words
    df_wc['words'] = list(set(df_wc['words'].str.split())) # count word only once per thesis
    df_wc = df_wc.explode('words')    

    # remove plurals
    import nltk
    from nltk.stem.wordnet import WordNetLemmatizer
    Lem = WordNetLemmatizer()
    df_wc['words'] = [Lem.lemmatize(plural) for plural in df_wc['words']]   

    # Add column for total number of times a word appears
    df_wc['total'] = df_wc.groupby('words')['words'].transform('count')

    # The number of times a word appears by year
    df_long = df_wc[['Year', 'words']].value_counts().to_frame('counts').reset_index()

    # Remove stopwords and punctuation
    if len(stop_words) > 0:
        df_long = df_long.loc[~df_long['words'].str.lower().isin(stop_words)]
    df_long['words'] = df_long['words'].str.replace('[{}]'.format(string.punctuation), '', regex=True)

    # Calculate the percentage of appearances per year
    df_long = df_long.merge(thesis_per_year, on='Year')
    df_long['percentage_of_appearances'] = 100 * df_long['counts'] / df_long['total_publications']

    return df_long

def preprocess_data(df, subjects=[], institutions=[], stop_words=[], word_column='Title', all_lower=True):
    """Preprocess data for animated bar charts.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with data to be plotted.
    subjects : str, optional
        Subjects to be plotted.
    stopwords : list, optional
        Words to be removed from the plot.
    word_coloumn : str, optional
        Column name of the dataframe that contains the words to be plotted.
    all_lower : bool, optional
        If True, all words are converted to lowercase.

    Returns
    -------
    pandas.DataFrame
        Preprocessed dataframe.
    """
    df_wc = filter_data(df, subjects, institutions)

    # Lowercase words
    if all_lower: 
        df_wc[word_column] = df_wc[word_column].str.lower()

    # Remove punctuation
    import string 
    df_wc['words'] = df_wc[word_column]
    df_wc['words'] = str(df_wc['words'])
    df_wc['words'] = df_wc['words'].str.replace('[{}]'.format(string.punctuation), '', regex=True)  
    df_wc['words'] = df_wc['words'].str.partition('\n')[0] # remove everything after the first line break
    
    # Number of theses per year
    thesis_per_year = df_wc['Year'].value_counts()
    thesis_per_year=thesis_per_year.to_frame('Count').reset_index()
    thesis_per_year.columns = ['Year', 'total_publications']    
    
    # Split words
    df_wc['words'] = df_wc['words'].str.split()
    df_wc = df_wc.explode('words')    

    # remove plurals
    import nltk
    from nltk.stem.wordnet import WordNetLemmatizer
    Lem = WordNetLemmatizer()
    df_wc['words'] = [Lem.lemmatize(plural) for plural in df_wc['words']]   

    # Add column for total number of times a word appears
    df_wc['total'] = df_wc.groupby('words')['words'].transform('count')

    # The number of times a word appears by year
    df_long = df_wc[['Year', 'words']].value_counts().to_frame('counts').reset_index()

    # Remove stopwords and punctuation
    if len(stop_words) > 0:
        df_long = df_long.loc[~df_long['words'].str.lower().isin(stop_words)]
    df_long['words'] = df_long['words'].str.replace('[{}]'.format(string.punctuation), '', regex=True)

    # Calculate the percentage of appearances per year
    df_long = df_long.merge(thesis_per_year, on='Year')
    df_long['percentage_of_appearances'] = 100 * df_long['counts'] / df_long['total_publications']

    return df_long

def plotly_animation(df, n_overall=50, n_yearly=10, percentage_per_year=False, by_variance=True):
    """Plot animated bar chart using plotly.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with data to be plotted.
    n_overall : int, optional
        Number of words to be plotted overall.
    n_yearly : int, optional
        Number of words to be plotted per year.
    percentage_per_year : bool, optional
        If True, words are plotted as percentage of appearances in theses per year.
    by_variance : bool, optional
        If True, words are selected based on variance in percentage of appearances per year (those which change the most over time selected). Else by the total number of appearances.

    Returns
    -------
    plotly.graph_objects.Figure
        Animated bar chart.
    """
    import pandas as pd
    import plotly.express as px

    if by_variance: 
        top_words = df.groupby('words')['percentage_of_appearances'].var().sort_values(ascending=False).head(n_overall).index
    else: 
        # Get the top n_overall unique words by total
        overall_top_words = df.groupby('words')['counts'].sum().sort_values(ascending=False).head(n_overall).index

        # Get the top n_yearly unique words by year
        yearly_top_words = df.groupby(['Year']).head(n_yearly)
        yearly_top_words = yearly_top_words['words'].unique()

        # Combined
        top_words = [value for value in overall_top_words if value in yearly_top_words]

    # data restructuring
    if percentage_per_year:
        column_name = 'percentage_of_appearances'
        plot_label = 'Percentage of theses the word<br>appearances in per year'
    else: 
        column_name = 'counts'
        plot_label = 'Appearances'

    range_max = max(df[column_name])

    #df['key'] = df.groupby(['Year','words']).cumcount()
    df = pd.pivot_table(df,index='Year', columns=['words'], values=column_name)
    df = df.stack(level=[0],dropna=False).reset_index()
    df.columns=['Year', 'words', 'counts']

    df = df.loc[df['words'].isin(top_words)]
    df['words'] = df['words'].astype('category')
    df['words'] = df['words'].cat.reorder_categories(top_words)

    # Plotting the data
    fig = px.bar(df, x="words", y="counts",  animation_frame="Year", range_y=[0, range_max])

    fig.update_layout(xaxis={'title': 'Word', 'visible': True, 'showticklabels': True}, yaxis={'title': plot_label})
    fig.update_xaxes(categoryorder='array', categoryarray= top_words)

    return fig


def line_plot_matplotlib(ax_input, df, n_appearances_threshold=50, n_words = 25, by_variance=True):
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # order by year
    df = df.sort_values(by=['Year'])

    keep_words = df.loc[df['cumulative_word_counts'] >= n_appearances_threshold, 'words']
    if len(keep_words) == 0:
        # error message if no words meet the threshold
        raise ValueError('No words meet the threshold of {} appearances, try reducing'.format(n_appearances_threshold))
    df = df.loc[df['words'].isin(keep_words) ]

    if by_variance: 
        # Get the words with the highest variance in ratio of appearances
        top_words = df.groupby('words')['ratio_of_appearances'].var().sort_values(ascending=False).head(n_words).index
    else: 
        # Get the most popular unique words (n_words)
        top_words = df.groupby('words')['counts'].sum().sort_values(ascending=False).head(n_words).index

    df = df.loc[df['words'].isin(top_words)]
    df['words'] = df['words'].astype('category')
    df['words'] = df['words'].cat.reorder_categories(top_words)

    df = pd.pivot(df, index=['Year'], columns = 'words', values = 'ratio_of_appearances') # Reshape from long to wide
    df = df.fillna(0)

    # replace zero with previous value in column
    df = df.replace(0, method='ffill')
    

    df.plot.line(ax=ax_input)
    ax_input.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 6})
    ax_input.set_xlabel('Year')
    ax_input.set_ylabel('Ratio of appearances')

def line_plot_seaborn(ax_input, df, n_appearances_threshold=50, n_words = 25, by_variance=True):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pylab as plt
    import matplotlib.ticker as ticker

    # order by year
    df = df.sort_values(by=['Year'])

    keep_words = df.loc[df['cumulative_word_counts'] >= n_appearances_threshold, 'words']

    if len(keep_words) == 0:
            # error message if no words meet the threshold
            raise ValueError('No words meet the threshold of {} appearances, try reducing'.format(n_appearances_threshold))
    df = df.loc[df['words'].isin(keep_words)]

    if by_variance:
        # Get the words with the highest variance in ratio of appearances
        top_words = df.groupby('words')['relative_frequency_of_word'].var().sort_values(ascending=False).head(n_words).index
    else:
        # Get the most popular unique words (n_words)
        top_words = df.groupby('words')['counts'].sum().sort_values(ascending=False).head(n_words).index

    df = df.loc[df['words'].isin(top_words)]
    df['words'] = df['words'].astype('category')
    df['words'] = df['words'].cat.reorder_categories(top_words)


    # line plot
    sns.set_style("darkgrid")
    #plt.xticks(rotation=-45, ha='left') 

    sns.lineplot(ax=ax_input, data=df, x='Year', y='relative_frequency_of_word', hue='words', palette='colorblind')

    # get n colour from the palette
    palette = sns.color_palette('colorblind', len(top_words))

    # # add text to the end of each line
    TEXTS = []
    for word in top_words:
        # get the value of the line when year is max
        max_year = df.loc[df['words'] == word, 'Year'].max()
        y = df.loc[(df['Year'] == max_year) & (df['words'] == word), 'relative_frequency_of_word'].values[0]
        colour = palette[top_words.get_loc(word)]
        TEXTS.append(ax_input.text(df['Year'].max(), y, word, horizontalalignment='left', verticalalignment='bottom', size='small', color=colour, weight='semibold'))

        #ax_input.text(df['Year'].max(), y, word, horizontalalignment='left', verticalalignment='bottom', size='small', color=colour, weight='semibold')

    from adjustText import adjust_text
    #from faker import Faker

    adjust_text(
        TEXTS, 
        only_move = {'points':'y', 'text':'y', 'objects':'y'},
        # expand_points=(2, 2),
        # arrowprops=dict(
        #     arrowstyle="->", 
        #     color='black', 
        #     lw=2
        # ),
        ax=ax_input #fig.axes[0]
    )

    ax_input.set(xlabel ="Year", ylabel = "Percentage of theses word has appeared in, to date")
    ax_input.legend(fontsize=8, bbox_to_anchor=(1.0, 1.01), loc='upper left')

    axis_jitter = 10/50*(df['Year'].max() - df['Year'].min())

    ax_input.set_xlim(df['Year'].min(), df['Year'].max()+axis_jitter)


def rolling_average(ax_input, df, n_appearances_threshold=50, n_words = 25, by_variance=True):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pylab as plt
    import matplotlib.ticker as ticker

    # order by year
    df = df.sort_values(by=['Year'])

    keep_words = df.loc[df['cumulative_word_counts'] >= n_appearances_threshold, 'words']

    if len(keep_words) == 0:
            # error message if no words meet the threshold
            raise ValueError('No words meet the threshold of {} appearances, try reducing'.format(n_appearances_threshold))
    df = df.loc[df['words'].isin(keep_words)]

    if by_variance:
        # Get the words with the highest variance in ratio of appearances
        top_words = df.groupby('words')['relative_frequency_of_word'].var().sort_values(ascending=False).head(n_words).index
    else:
        # Get the most popular unique words (n_words)
        top_words = df.groupby('words')['counts'].sum().sort_values(ascending=False).head(n_words).index

    df = df.loc[df['words'].isin(top_words)]
    df['words'] = df['words'].astype('category')
    df['words'] = df['words'].cat.reorder_categories(top_words)


    # line plot
    sns.set_style("darkgrid")
    #plt.xticks(rotation=-45, ha='left') 

    sns.lineplot(ax=ax_input, data=df, x='Year', y='rolling_average', hue='words', palette='colorblind')

    # get n colour from the palette
    palette = sns.color_palette('colorblind', len(top_words))

    # # add text to the end of each line
    TEXTS = []
    for word in top_words:
        # get the value of the line when year is max
        max_year = df.loc[df['words'] == word, 'Year'].max()
        y = df.loc[(df['Year'] == max_year) & (df['words'] == word), 'rolling_average'].values[0]
        colour = palette[top_words.get_loc(word)]
        TEXTS.append(ax_input.text(df['Year'].max(), y, word, horizontalalignment='left', verticalalignment='bottom', size='small', color=colour, weight='semibold'))

        #ax_input.text(df['Year'].max(), y, word, horizontalalignment='left', verticalalignment='bottom', size='small', color=colour, weight='semibold')

    from adjustText import adjust_text
    adjust_text(
        TEXTS, 
        only_move = {'points':'y', 'text':'y', 'objects':'y'},
        ax=ax_input
    )

    ax_input.set(xlabel ="Year", ylabel = "Percentage of theses word has appeared in, to date")
    ax_input.legend(fontsize=8, bbox_to_anchor=(1.0, 1.01), loc='upper left')

    axis_jitter = 10/50*(df['Year'].max() - df['Year'].min())

    ax_input.set_xlim(df['Year'].min(), df['Year'].max()+axis_jitter)