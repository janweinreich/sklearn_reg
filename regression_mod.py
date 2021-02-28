from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

class KRR(BaseEstimator, RegressorMixin):

    def __init__(self, kernel_func , sigma=None, lamb=None, kpca_ncmps=None, bst_mod = None):

        """
        User must provide a kernelfunction, 
        sigma and lambda are optinal for initialization
        """

        self.kernel_func  = kernel_func        # kernelfunction, no default
        self.sigma        = sigma or 1.0       # sigma value, 1 default
        self.lamb         = lamb  or 1e-7      # lambda value, 1e-7 default
        self.kpca_ncmps   = kpca_ncmps or 2    # number of components for kpca, 2 default
        self.bst_mod      = bst_mod or 0   
    
        
        
        
        """
        get_params and set_params are used by sklearn e.g. for hyperparameter
        optimization
        """

    def get_params(self, deep=True):

        return {"kernel_func":self.kernel_func,"sigma": self.sigma, "lamb": self.lamb}

    def set_params(self, **parameters):

        self.kernel_func   = parameters["kernel_func"]
        self.sigma         = parameters["sigma"]
        self.lamb          = parameters["lamb"]      
        
        
        return self

        """
        Obtain the regression coefficients alpha and
        save the training representations because they are 
        needed for predictions
        """
    
    def fit(self, X, y):

        Kernel_train  = self.kernel_func(X,X, self.sigma)

        for i in range(Kernel_train.shape[0]):
            Kernel_train[i, i] += self.lamb

   
        self.ALPHA_SOL = np.linalg.solve(Kernel_train, y)
        self.XTRAIN    = X
        
        return self
    
        """
        Make the predictions for query representations Xry
        """

    def predict(self, Xqry):

        Kernel_test     = self.kernel_func(self.XTRAIN, Xqry, self.sigma)
        pred            = np.dot(self.ALPHA_SOL, Kernel_test)
        
        return pred
    
        """
        Make a kernel-pca, by first going to kernel space and 
        performing a pca afterwards, by default 
        returns the first two components
        """



    """
    def cv_score(self, X, y):

        from sklearn.model_selection import cross_val_score
        kfold = 5
        if self.bst_mod==0:
           print("No model to validate") 
           exit()


        scores = cross_val_score(KRR(self.bst_mod.get_params()['krr__kernel_func'],self.bst_mod.get_params()['krr__sigma'],self.bst_mod.get_params()['krr__lamb']), X, y, scoring='neg_mean_absolute_error', cv=kfold)

        mae_scores = np.array([abs(s) for s in scores])
        mean_score = np.mean(mae_scores)
        print('MAE: {:.3f}'.format(mean_score))
        return mae_scores
    """



    def kpca(self, X):
        Kernel_train  = self.kernel_func(X,X, self.sigma)

        from sklearn.decomposition import PCA

        pca     = PCA(n_components=self.kpca_ncmps)
        kpca    = pca.fit_transform(Kernel_train)

        return kpca