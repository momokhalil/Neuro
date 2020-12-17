# Imports
import numpy as np
import matplotlib.pyplot as plt
from neuro.models import Sequential as Seq
from neuro.layers import Dense, Predict


# generate n real samples with class labels
def generate_real_samples(n):
    X1 = 5 * (np.random.rand(n, 1) - 0.5)
    X2 = np.sin(2*X1)
    X = np.hstack((X1, X2))
    y = np.ones((n, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim_, batch_):
    return np.random.randn(batch_, latent_dim_)


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(gen_, latent_dim_, batch_):
    x_input = generate_latent_points(latent_dim_, batch_)
    return gen_.forward(x_input), np.zeros((batch_, 1))


# Generate data for training discriminator
def generate_disc_training_data(generator_, latent_dim_, batch_):
    x_r, y_r = generate_real_samples(batch_ // 2)
    x_f, y_f = generate_fake_samples(generator_, latent_dim_, batch_ // 2)
    return np.vstack((x_r, x_f)), np.vstack((y_r, y_f))


# Generate data for training generator
def generate_gan_training_data(latent_dim_, batch_):
    return generate_latent_points(latent_dim_, batch_), np.ones((batch_, 1))


# Build all models
def build_model(l2_, latent_dim_):
    # Generator Sequential Model
    generator_ = Seq(Dense(30, latent_dim_, l2_, activation_type='relu'),
                     Dense(30, 30, l2_, activation_type='relu'),
                     Dense(2, 30, l2_, activation_type='linear'))

    # Discriminator Sequential Model
    discriminator_ = Seq(Dense(30, 2, l2_, activation_type='relu'),
                         Dense(30, 30, l2_, activation_type='relu'),
                         Predict(1, 30, l2_, activation_type='sigmoid'))
    discriminator_.build(loss_type='binary_cross_entropy')

    # Generator Adversarial Network Sequential Model
    gan_ = Seq(generator_, discriminator_)
    gan_.build(loss_type='binary_cross_entropy')

    return generator_, discriminator_, gan_


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Set general parameters
    l2 = 0.005
    latent_dim = 5
    batch = 256

    # Build model
    generator, discriminator, gan = build_model(l2, latent_dim)

    # Animation
    plt.ion()

    fig = plt.figure(figsize=(8, 6))
    real, = plt.plot(0, 0, marker='o', color='black', ls='')
    fake, = plt.plot(0, 0, marker='o', color='blue', ls='')
    plt.ylim(-2, 2)
    plt.xlim(-3, 3)

    # Hyper parameters
    alpha = 0.0024
    decay = 0.99995

    # Perform iterations
    for step in range(15000):

        # Discriminator Training
        x_d, y_d = generate_disc_training_data(generator, latent_dim, batch)
        discriminator.trainable = True
        discriminator.fit(x_d, y_d, iter_epoch=1, alpha=alpha, decay=decay)

        # Generator Training
        x_g, y_g = generate_gan_training_data(latent_dim, batch)
        discriminator.trainable = False
        gan.fit(x_g, y_g, iter_epoch=1, alpha=alpha, decay=decay)

        # Plot prediction
        if step % 400 == 0:
            real.set_data(x_d[:batch//2, 0], x_d[:batch//2, 1])
            fake.set_data(x_d[batch//2:, 0], x_d[batch//2:, 1])
            plt.show()
            plt.pause(0.01)

    plt.waitforbuttonpress()
