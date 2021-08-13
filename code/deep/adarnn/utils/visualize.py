# A wrapper of Visdom for visualization

import visdom
import numpy as np
import time

class Visualize(object):
    def __init__(self, port=8097, env='env'):
        self.port = port
        self.env = env
        self.vis = visdom.Visdom(port=self.port, env=self.env)

    def plot_line(self, Y, global_step, title='title', legend=['legend']):
        """ Plot line
        Inputs:
            Y (list): values to plot, a list
            global_step (int): global step
        """
        y = np.array(Y).reshape((1, len(Y)))
        self.vis.line(
            Y = y, 
            X = np.array([global_step]), 
            win = title,
            opts = dict(
                title = title,
                height = 360,
                width = 400,
                legend = legend,
            ),
            update = 'new' if global_step==0 else 'append'
        )

    def heat_map(self, X, title='title'):
        self.vis.heatmap(
            X = X,
            win = title,
            opts=dict(
                title = title,
                width = 360,
                height = 400,
            )
        )

    def log(self, info, title='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        """

        log_text = ('[{time}] {info} <br>'.format(
                            time=time.strftime('%m%d_%H%M%S'),\
                            info=info)) 
        self.vis.text(log_text, title, append=True)  


if __name__ == '__main__':
    vvv = visdom.Visdom(port=8097, env='test')
    def test():
        vis = Visualize(env='test')
        import time
        for i in range(10):
            y = np.random.rand(1, 2)
            title = 'Two values'
            legend = ['value 1', 'value 2']
            vis.plot_line([y[0,0], y[0,1]], i, title, legend)
            vvv.line(Y=np.array([y[0,0], y[0,1]]).reshape((1,2)), X=np.array([i]),win='test2', update='append')
            time.sleep(2)
    test()