import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the prior distribution
    
    Outputs: None
    -----
    """
    
    # N(mu_of_a, co_variance_of_a)
    mu_of_a = np.zeros((2,), dtype=float)
    co_variance_of_a = beta * np.eye(2)

    # Create a grid of points for plotting the contours
    a0_val, a1_val = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
    a0_mesh, a1_mesh = np.meshgrid(a0_val, a1_val)

    # Combine the grid points into a single 2D array of samples
    samples = np.concatenate((a0_mesh.reshape(-1, 1), a1_mesh.reshape(-1, 1)), axis=1)

    # Compute the Gaussian density
    prior_density = util.density_Gaussian(mu_of_a, co_variance_of_a, samples)

    # Reshape the density into a 2D array
    prior_density = prior_density.reshape((100, 100))

    # Plot the contours and colorbar
    fig, ax = plt.subplots()
    ax.plot([-0.1], [-0.5], marker='o', markersize=7, color='orange')
    
    # Set the labels and title
    ax.set_xlabel('a0')
    ax.set_ylabel('a1')
    ax.set_title('Prior distribution: a0 vs a1')
    
    #Add the colorbar to the contour plot
    ax.contourf(a0_mesh, a1_mesh, prior_density, cmap='viridis')
    
    # Add the colorbar label
    cbar = fig.colorbar(ax.contourf(a0_mesh, a1_mesh, prior_density, cmap='viridis'))
    cbar.ax.set_ylabel('Density')

    # Save the plot and show it
    plt.savefig('prior.pdf')
    plt.show()

    return

def posteriorDistribution(x, z, beta, sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    # Calculate the posterior distribution parameters
    N = x.shape[0]
    design_matrix = np.column_stack((np.ones(N), x.ravel()))
    inv_prior_cov = (beta / sigma2) * np.eye(2)
    posterior_cov = np.linalg.inv(inv_prior_cov + (1 / sigma2) * design_matrix.T @ design_matrix)
    posterior_mean = (1 / sigma2) * posterior_cov @ design_matrix.T @ z

    # Plot the posterior distribution
    plot_posterior_distribution(x, z, beta, sigma2, posterior_mean, posterior_cov)

    return posterior_mean.ravel(), posterior_cov

def plot_posterior_distribution(x, z, beta, sigma2, mu, Cov):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from training set
    beta: hyperparameter in the prior distribution
    sigma2: variance of Gaussian noise
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """

    # Create a grid of points for plotting the contours
    x_mesh, z_mesh = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))

    # Compute the Gaussian density for the grid points
    x_val = np.column_stack((x_mesh.ravel(), z_mesh.ravel()))
    
    # post_Prob defination
    post_prob = util.density_Gaussian(mu.T, Cov, x_val).reshape(x_mesh.shape)

    # Set the style of the plot
    plt.style.use('seaborn-darkgrid')
    plt.rcParams['font.family'] = 'Arial'

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the contours
    cf = ax.contourf(x_mesh, z_mesh, post_prob, cmap='Blues')

    # Plotting the point
    ax.plot([-0.1], [-0.5], marker='o', markersize=7, color='orange')

    # Set the labels and title
    ax.set(xlabel='a0', ylabel='a1', title="Posterior for N = %d" % len(x))

    # Set the limits of the axes
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    # Add the colorbar label
    cbar = fig.colorbar(cf, ax=ax)
    cbar.ax.set_ylabel('Density', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Set the font size of the tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Save the plot
    plt.savefig("posterior%d.pdf" % len(x))

    # Show the plot
    plt.show()



def predictionDistribution(x, beta, sigma2, mu, Cov, x_train, z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the prior distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train, z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    N = np.size(x_train, 0)
    fig, ax = plt.subplots()
        
    for i, xi in enumerate(x):
        
        # Calculate the covariance of the posterior distribution
        cov_test = Cov[0, 0] + 2*Cov[0, 1]*xi + Cov[1, 1]*xi**2
        standard_deviation_test = np.sqrt(cov_test)
        
        # Calculate the mean of the posterior distribution
        mu_test = mu[0] + mu[1]*xi
        
        # Plot the mean and standard deviation of the posterior distribution
        test_sample = plt.errorbar(xi, mu_test, yerr = standard_deviation_test, color = 'goldenrod', 
                                ecolor = 'maroon', elinewidth = 3, capsize = 0, marker = 'o', markersize = 2)


    # Plot the training samples used in the calculation of the posterior distribution
    train_sample = ax.scatter(x_train[:, 0], z_train[:, 0], color='red', s=7)
        
    # Set the limits of the axes
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    
    
    # Set the labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Prediction with N = %d Training Samples' % N)  
    
    # Add a legend
    ax.legend([test_sample, train_sample], ['Test ', 'Training '],
              loc='upper left')
    
    # Save the plot
    plt.savefig("predict%d.pdf" % N)
    plt.show()
    return 






if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 100
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
