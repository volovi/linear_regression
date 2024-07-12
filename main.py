import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RangeSlider
from sklearn.datasets import make_regression

N = 1
M = 50

min_learning_rate, init_learning_rate, max_learning_rate = 0.1, 0.5, 2.0
min_epochs, init_epochs_from, init_epochs_to, max_epochs = 0, 1, 12, 15

init_epochs = (init_epochs_from, init_epochs_to)
init_epochs_pre = init_epochs_from - min_epochs
init_epochs_post = max_epochs - init_epochs_to

cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()['color']
marker_sizes = ((np.eye(max_epochs + 1) * 2) + 1) * 36


# def f(w, b):
#     return [[cost(_x * w[i,j] + b[i,j], _y) for j in range(M)] for i in range(M)]


def f(w, b):
    return np.mean([0.5 * (x * w + b - y) ** 2 for x, y in zip(_x, _y)], axis=0)


def cost(a, y):
    return 0.5 * np.mean((a - y) ** 2)


def cost_derivative(a, y):
    return a - y


def gradient(dj_da, x):
    dj_dw = (1 / M) * x.T @ dj_da
    dj_db = (1 / M) * np.sum(dj_da, axis=0, keepdims=True)
    
    return dj_dw, dj_db


def params(learning_rate):
    w = np.full((N, 1), 1e-6)
    b = np.zeros((1, 1))
    p = np.empty((0, 3))

    for epoch in range(max_epochs + 1):
        a = _x @ w + b
        p = np.vstack((p, (w.squeeze(), b.squeeze(), cost(a, _y))))
            
        dj_dw, dj_db = gradient(cost_derivative(a, _y), _x)
        w -= learning_rate * dj_dw
        b -= learning_rate * dj_db

    return p.T


def lines_data(x_min, x_max, y_min, y_max, w, b):
    xs1 = (y_min - b) / w
    xs2 = (y_max - b) / w
    
    xs = np.array((np.minimum(np.maximum(x_min, xs1), np.maximum(x_min, xs2)),
                   np.maximum(np.minimum(x_max, xs2), np.minimum(x_max, xs1))))
    
    return np.dstack((xs.T, (xs * w + b).T))


def limits(x_min, x_max, y_min, y_max):
    xm = 0.05 * (x_max - x_min)
    ym = 0.05 * (y_max - y_min)

    return dict(xlim=(x_min - xm, x_max + xm), ylim=(y_min - ym, y_max + ym))


def limits2(w, b, j):
    i = np.argpartition(j, 1)[:2]

    wm = np.median(w[i])
    bm = np.median(b[i])

    r = np.max((np.max(w) - wm, wm - np.min(w), np.max(b) - bm, bm - np.min(b)))

    return limits(wm - r, wm + r, bm - r, bm + r)


def contours(xlim, ylim):
    w, b = np.meshgrid(np.linspace(*xlim), np.linspace(*ylim))
    cs = ax2.contour(w, b, f(w, b), levels=19, linewidths=0.5)
    cl = ax2.clabel(cs, inline=True, fontsize=10)
    
    return cs, cl


def setup1(w, b, j):
    ax1.set(**limits(*_l))

    lines = ax1.plot(*np.empty((2, 0, init_epochs_pre)),
                     *np.c_[lines_data(*_l, w, b)].T,
                     *np.empty((2, 0, init_epochs_post)),
                     linestyle=':', marker=(4, 1), linewidth=0.5)

    ls = lines[init_epochs_from : init_epochs_to + 1]

    ls[np.argmin(j)].set(markersize=10.0, linestyle='-')
    ax1.legend(ls, j.round(1), loc=2)

    values = ax1.scatter(_x, _y)

    return lines, values


def setup2(w, b, j):
    lim = limits2(w, b, j)
    
    ax2.set(**lim) 
    cs, cl = contours(**lim)

    path, = ax2.plot(w, b, ':', linewidth=0.5)
    patho = ax2.scatter(w, b, zorder=2, marker=(4, 1))

    patho.set_color(np.roll(cycle_colors, -init_epochs_from))
    patho.set_sizes(marker_sizes[np.argmin(j)])

    return [cs, cl], path, patho


def update1(w, b, j):
    ls = lines[ep_sl.val[0] : ep_sl.val[1] + 1]

    for line, data in zip(ls, lines_data(*_l, w, b)):
        line.set(data=data.T, markersize=6.0, linestyle=':')

    ls[np.argmin(j)].set(markersize=10.0, linestyle='-')
    ax1.legend(ls, j.round(1), loc=2)


def update2(w, b, j):
    path.set_data(w, b)
    patho.set_offsets(np.c_[w, b])

    patho.set_sizes(marker_sizes[np.argmin(j)])


def lr_changed(val):
    _p[:] = params(val)

    update1(*p())
    update2(*p())


_v = [init_epochs] # a workaround for the RangeSlider bug
def ep_changed(val):
    if _v[0] == val:
        return
    _v[0] = val

    patho.set_color(np.roll(cycle_colors, -val[0]))

    for i, line in enumerate(lines):
        line.set_visible(i >= val[0] and i <= val[1])

    update1(*p())
    update2(*p())


def relim(event):
    for obj in _q[0].collections:
        obj.remove()

    for obj in _q[1]:
        obj.remove()

    ax1.set(**limits(*_l))
    lim = limits2(*p())
    
    ax2.set(**lim)
    _q[:] = contours(**lim)

    fig.canvas.draw_idle()


def reset(event):
    lr_sl.reset()
    ep_sl.reset()


def rand(event):
    x, y = make_regression(n_samples=M, n_features=N, noise=10)
    
    _x[:, 0] = x.squeeze()
    _y[:, 0] = y
    _l[:] = np.min(_x), np.max(_x), np.min(_y), np.max(_y)

    values.set_offsets(np.c_[_x, _y])

    lr_sl.set_val(init_learning_rate) # forced update
    ep_sl.reset()

    relim(0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.7))

ax3 = fig.add_axes([0.124, 0.15, 0.777, 0.03])
ax4 = fig.add_axes([0.124, 0.1, 0.777, 0.03])
ax5 = fig.add_axes([0.6, 0.02, 0.09, 0.04])
ax6 = fig.add_axes([0.72, 0.02, 0.09, 0.04])
ax7 = fig.add_axes([0.84, 0.02, 0.09, 0.04])

lr_sl = Slider(ax3, 'α', min_learning_rate, max_learning_rate, valinit=init_learning_rate, valstep=0.001)
lr_sl.on_changed(lr_changed)

ep_sl = RangeSlider(ax4, 'epochs', min_epochs, max_epochs, valinit=init_epochs, valstep=1)
ep_sl.on_changed(ep_changed)

relim_bt = Button(ax5, 'Relim')
relim_bt.on_clicked(relim)

reset_bt = Button(ax6, 'Reset')
reset_bt.on_clicked(reset)

rand_bt = Button(ax7, 'Rand')
rand_bt.on_clicked(rand)

_x, _y = make_regression(n_samples=M, n_features=N, noise=10, random_state=False)
_y = _y[:, None]
_l = [np.min(_x), np.max(_x), np.min(_y), np.max(_y)]

_p = params(init_learning_rate)

p = lambda: _p[:, ep_sl.val[0] : ep_sl.val[1] + 1]

lines, values = setup1(*p())
_q, path, patho = setup2(*p())

plt.subplots_adjust(bottom=0.25)
plt.show()

