red = '#F8423E'
orange = '#FF8000'
green = '#2EB42E'
blue = '#0F7FFD'
purple = '#C837F5'
pink = '#F71480'
colors = {'red': red, 'orange': orange, 'green': green, 'blue': blue, 'purple': purple, 'pink': pink}


def set_style(ax, **kwargs):
    ax.tick_params(axis='y', length=4, width=1.5, pad=6, reset=True, which='major', bottom=False, top=True, left=True,
                   right=False)
    ax.tick_params(axis='x', length=4, width=1.5, pad=6, reset=True, which='major', bottom=True, top=False, left=False,
                   right=False)
    ax.grid(False)

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    return ax
