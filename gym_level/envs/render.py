import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

style.use('dark_background')


def date2num(date):
    converter = mdates.strpdate2num('%Y-%m-%d')
    return converter(date)


class StatePlotter:
    """A stock trading visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, title=None, num_states=1, num_actions=1):
        self.states = np.zeros((num_states, 50))
        self.actions = np.zeros((num_actions, 50))

        fig = plt.figure()
        fig.suptitle(title)

        num_cols = max(num_states, num_actions)
        self.ax_state, self.ax_action = [], []
        for i in range(num_states):
            self.ax_state.append(fig.add_subplot(2, num_cols, i+1))

        for i in range(num_actions):
            self.ax_action.append(fig.add_subplot(2, num_cols, num_cols + i+1))

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0.2)

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def render_value(self, current_step, state, i, type_='states'):
        if type_ == 'states':
            states = self.states
            axis = self.ax_state[i]
        else:
            states = self.actions
            axis = self.ax_action[i]

        states[i, current_step] = state

        # Clear the frame rendered last step
        axis.clear()

        # Plot net worths
        axis.plot(states[i, :], '-', label=type_)

        # Show legend, which uses the label we defined for the plot above
        axis.legend()
        legend = axis.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        # Annotate the current net worth on the net worth graph
        axis.annotate('{0:.2f}'.format(state), (current_step, state),
                                   xytext=(current_step, state),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        # Add space above and below min/max net worth
        min_ = min(states[i, :])
        max_ = max(states[i, :])
        diff_ = max_ - min_
        axis.set_ylim(min_ - diff_/10, max_ + diff_/10)

    def render(self, current_step, state, action):
        for i in range(state.shape[0]):
            self.render_value(current_step, state[i], i, 'states')
        for i in range(action.shape[0]):
            self.render_value(current_step, action[i], i, 'actions')

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()