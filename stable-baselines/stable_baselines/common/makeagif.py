import imageio
import numpy as np

def makeagif(model, title, save_path):
    images = []
    obs = model.env.reset()
    img = model.env.render(mode='rgb_array')
    for i in range(500):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _, _ = model.env.step(action)
        img = model.env.render(mode='rgb_array')

    imageio.mimsave(save_path + title + '.gif', [np.array(img[0]) for i, img in enumerate(images) if i % 2 == 0], fps=29)
