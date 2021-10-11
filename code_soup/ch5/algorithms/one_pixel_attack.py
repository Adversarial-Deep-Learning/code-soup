import numpy as np
import torch
from scipy.optimize import differential_evolution

"""
The code is a slight modification and refactoring from the following repository
https://github.com/Hyperparticle/one-pixel-attack-keras

"""


class OnePixelAttack:

    """
    Attack using One Pixel

    """

    def __init__(self, model, device=None):

        """

        Initialize the Attack obj
        Parameters
        ----------
        model: torch.nn
            model to be attacked
        device : string
            'cpu' / 'cuda'

        """

        self.model = model
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def perturb_image(self, perturbation, orig_img):  # pragma: no cover
        """
        Parameters
        ----------
        orig_image: image to be changed
        perturbation: tuple of (x,y,r,g,b) on basis of which the image is changed
                      [(),(),()]. Image can have multiple perturbations
        """
        x_pos, y_pos, *rgb = perturbation
        new_img = orig_img
        new_img[:, x_pos, y_pos] = rgb

        return new_img

    def perturbation_image(self, perturbation_array, image):  # pragma: no cover
        """
        Applies multiple perturbation to a single image
        Parameters
        ----------
        image: the image to be perturbated
        perturbation_array: array like [x1, y1, r1, g1, b1, x2, y2, r2, g2, b2, ...]
                            the number of pixels to be changed
        """
        perturbation_array = np.array(perturbation_array)

        perturbation_array = perturbation_array.astype(int)

        perturbation_array = np.split(perturbation_array, len(perturbation_array) // 5)

        new_image = image

        for perturb in perturbation_array:
            new_image = self.perturb_image(perturb, new_image)

        return new_image

    def predict_class(self, xs, img, target_class, minimize=True):  # pragma: no cover
        """
        Parameters
        ----------
        xs: 1d array to be evolved
        img: image to be perturbed
        target_class: class to be targeted or avoided
        minimize: This function should always be minimized, so return its complement
                if needed basically is targetted increase the prob
        """
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        img_perturbed = self.perturbation_image(xs, img)
        prediction = self.model_predict(img_perturbed)[target_class]

        # This function should always be minimized, so return its complement
        # if needed basically is targetted increase the prob
        return prediction if minimize else 1 - prediction

    def model_predict(self, image):  # pragma: no cover
        """
        Helper function to predict probs from the model of just 1 image
        """
        prediction = None
        with torch.no_grad():
            image = torch.FloatTensor(image).reshape(1, *image.shape)
            image = image.to(self.device)
            prediction = self.model(image)[0].detach().numpy()
        return prediction

    def attack_success(
        self, x, img, target_class, targeted_attack=False, verbose=False
    ):  # pragma: no cover
        """
        check if the attack is a success. the callback helper function for differential_evolution
        Parameters
        ----------
        x: 1d array that is evolved
        img: image to be perturbed
        target_class: class to be targeted or avoided
        Returns:
        -------
        true if the evolution needs to be stopped
        """
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = self.perturbation_image(x, img)

        confidence = self.model_predict(attack_image)
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or
        # targeted classification), return True
        if verbose:
            print("Confidence:", confidence[target_class])
        if (targeted_attack and predicted_class == target_class) or (
            not targeted_attack and predicted_class != target_class
        ):
            return True

        return False

    def attack(
        self,
        image,
        original_label,
        target=None,
        pixel_count=1,
        maxiter=75,
        popsize=400,
        verbose=False,
    ):  # pragma: no cover

        """
        Runs the attack on a single image, searches the image space
        Parameters
        ----------
        image: Numpy.Array
            image of shape(channel, height, width)
        original_label: int
            class the image belongs too
        target: int
            class to be targetted
        pixels_count: int
            Number of Pixels to be perturbed (changed)
        maxiter:int, optional
            The maximum number of generations over which the entire population is evolved.
            The maximum number of function evaluations (with no polishing) is: (maxiter + 1) * popsize * len(x)
        popsize:int, optional
            A multiplier for setting the total population size. The population has popsize * len(x) individuals.
        verbose: boolean
            set to true to print the confidence
        Returns
        -------
         perturbation_array:
            List of all the best perturbations to the images in the batch


        """

        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else original_label

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        print("Image shape:", image.shape)
        dim_x, dim_y = image.shape[1], image.shape[2]
        bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            print("Predicting!")

            return self.predict_class(xs, image, target_class, target is None)

        def callback_fn(x, convergence):
            print("check success!")
            return self.attack_success(x, image, target_class, targeted_attack, verbose)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn,
            bounds,
            maxiter=maxiter,
            popsize=popmul,
            recombination=1,
            atol=-1,
            callback=callback_fn,
            polish=False,
        )

        # -----------------  Calculate some useful statistics to return from this function
        # attack_image = self.perturbation_image(attack_result.x, image)
        # prior_probs = self.model_predict(image)
        # predicted_probs = self.model_predict(attack_image)
        # predicted_class = np.argmax(predicted_probs)
        # actual_class = original_label
        # success = predicted_class != actual_class
        # cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        # Show the best attempt at a solution (successful or not)
        # if plot:
        #     helper.plot_image(attack_image, actual_class, self.class_names, predicted_class)

        # return [pixel_count, attack_image, actual_class, predicted_class, success, cdiff, prior_probs,
        #         predicted_probs, attack_result.x]

        # return the best perturbation array
        return attack_result.x

    def step(
        self,
        data,
        labels=None,
        pixels_perturbed=1,
        targeted=False,
        maxiter=75,
        popsize=400,
        verbose=False,
    ):
        """
        Runs the attack on a batch of images, searches the image space on a single batch of images
        Parameters
        ----------
        data : torch.Tensor
            Batch of data
        labels: List
            list of all the unique classes from the dataset
        pixels_perturbed: int
            Number of Pixels to be perturbed (changed)
        targeted: boolean
            To decide if this is a targetted attack or not (in casee of targetted attack run all labels)
        maxiter:int, optional
            The maximum number of generations over which the entire population is evolved.
            The maximum number of function evaluations (with no polishing) is: (maxiter + 1) * popsize * len(x)
        popsize:int, optional
            A multiplier for setting the total population size. The population has popsize * len(x) individuals.
        verbose: boolean
            set to true to print the confidence

        Returns
        -------
         perturbation_array:
            List of all the best perturbations to the images in the batch

        """

        images, image_orig_label = data
        batch_size = len(images)

        # store the best perturbation for all the images in the batch
        perturbation_array = []

        for i in range(batch_size):
            image = images[i].detach().numpy()
            orig_label = image_orig_label[i].detach().numpy().astype(int)
            targets = [None] if not targeted else range(len(labels))

            for target in targets:
                if targeted:  # pragma: no cover
                    print("Attacking with target", labels[target])
                    if target == orig_label:
                        continue
                best_perturbation = self.attack(
                    image,
                    orig_label,
                    target,
                    pixels_perturbed,
                    maxiter=maxiter,
                    popsize=popsize,
                    verbose=verbose,
                )
                perturbation_array.append(best_perturbation)

        return perturbation_array
