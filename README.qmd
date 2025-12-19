---
title: "Demonstrating Quarto for GitHub READMEs"
format: gfm
execute: 
  warning: false
  message: false
  errors: false
jupyter: python3
---

## Section Header

You'll want to make use of sections and subsections to organize your document. Think of the sections as broad categories, and subsections as more specific topics within those categories. I can make bigger statements at this level.

### Subsection Header

Subsections are useful for breaking ideas down and adding details. This is also likely where you'll want to add code blocks. 

It will be worth noting some chunk options that you might find useful.

If I want to hide code but show output, I can use `echo: false`.

```{python}
#| echo: false
2 + 2
```

If I want to hide output but show code, I can use `results: 'hide'`.

```{python}
#| results: 'hide'
2 + 2
```

Maybe I want to show the code, but not run it. I can use `eval: false`.

```{python}
#| eval: false
2 + 2
```

In most of your chunks, you'll probably want to turn off messages and warnings. You might notice, though, that I did that globally in the execute key in the header above. But if you want to do it in a specific chunk, you can use:

```{python}
#| messages: false
#| warnings: false
```

## Loading Data and Models

If your project relies on heavy computations (either for the data prep or modeling), you'll probably want to show your work in a code chunk that does not evaluate. 

```{python}
#| eval: false
#| echo: true

'''
Imagine that this is a long code block that reads data
and then starts your wrangling tasks. If it takes 
over a minute to run, you'll probably want to have
that work saved in a script and write your data out
to a file.
'''

```

You can quietly load in your prepped data in another chunk.

```{python}
#| echo: false
import pandas as pd
#data = pd.read_csv('path/to/your/prepped_data.csv')
```

The same holds true for models, especially if you're fitting complex models that take 

## Rendering Outputs

When I render this document, the output will be ready to appear as the `readme` on my GitHub repository. Let's see how that might look with some plots and tables.

Notice below that I'm not going to print my code, I just want it to run.
```{python}
#| echo: false
#| messages: false
#| warnings: false

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
```

Let's see what a simple plot looks like.

```{python}
#| messages: false
#| warnings: false

data = {
    'x': np.linspace(0, 10, 100),
    'y': np.sin(np.linspace(0, 10, 100))
}

df = pd.DataFrame(data)

sns.lineplot(data=df, x='x', y='y')
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

This is where I am going to offer some explanations about the sine wave above. 

Now let's see what a table looks like.

```{python}
#| echo: true
#| messages: false
#| warnings: false
#| results: 'markdown'
data_table = {
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e'],
    'C': [10.5, 20.3, 30.2, 40.1, 50.0]
}
df_table = pd.DataFrame(data_table)
df_table.head().to_markdown()
```

Notice the use of `to_markdown()` to render the table nicely in GitHub's markdown format. It is a small thing that will make a big in difference in how your document looks!

Now, let's see what some linear model output looks like.

```{python}
#| echo: true
#| messages: false
#| warnings: false
import statsmodels.api as sm    
X = df[['x']]
y = df['y']
X = sm.add_constant(X) 
model = sm.OLS(y, X).fit()
summary_table = model.summary().tables[1] 
print(summary_table)
```

Notice how I indexed into the summary table object to just get the coefficients table?

You migth want to talk about your coefficients and there are two ways that you can do that:

1. Manually type out your interpretations below the table.
2. Use code to extract the coefficients and generate text automatically.

Option number 1, while more straightforward, is not the choice. Option number 2 is more dynamic and reproducible, especially if you plan to update your model or data in the future.

```{python}
#| echo: true
#| messages: false
#| warnings: false
params = model.params
intercept = params['const']
slope = params['x'] 
```

Now that those are out, I can use inline chunks to reference them in my text. You can think of inline chunks a lot like f-strings.

The intercept of the model is `{python} str(intercept.round(3))` and the slope is `{python} str(slope.round(3))`. No matter what I do with my data or model, these values will always be up to date in my text!

I can also pull out model fit statistics like *R*-squared. The *R*-squared for this model is `{python} str(model.rsquared.round(3))`.

You have the full power of markdown and Python at your disposal. I just want to show you some handy markdown things that you might want to use. 

I can create **bold** text or *italicized* text.

Functions and variables can be represented in `monospace`. Note that those are backticks, not apostrophes.

You can create bullet point lists:

- Item 1
- Item 2
  - Subitem 2a
  - Subitem 2b

You might want to include links within your document. You can use html links like this: <a href="https://www.example.com">Example</a> or markdown links like this: [Example](https://www.example.com).