library(xgboost)
dml_xgboost = function(y, d, x, 
                       k_fold = 5, 
                       k_fold_validation = 10,
                       y.params = list(),
                       d.params = list(),
                       verbose = 0){
  n = nrow(x)
  fold_id = c(rep(1:k_fold, n/k_fold), 1:((n/k_fold-as.integer(n/k_fold))*k_fold))
  folds = data.frame(i = 1:n, f = sample(fold_id, size = n))
  
  fits = list()
  y.pred = list()
  d.pred = list()
  
  for (k in 1:k_fold) {
    train_index = folds$i[folds$f != k]
    test_index = folds$i[folds$f == k]
    x.train = as.matrix(x[train_index,])
    y.train = as.matrix(y[train_index])
    d.train = as.matrix(d[train_index])
    
    x.test = as.matrix(x[test_index,])
    y.test = as.matrix(y[test_index])
    d.test = as.matrix(d[test_index])
    
    xgbCV = xgb.cv(data = x.train, label = y.train, nfold = k_fold_validation, early_stopping_rounds = 5, nrounds = 1000, params = y.params, verbose = verbose)
    xgbCV$best_iteration
    xgb.y = xgboost(data = x.train, label = y.train, nrounds = xgbCV$best_iteration, verbose = verbose, params = y.params)
    
    xgbCV = xgb.cv(data = x.train, label = d.train, nfold = 5, early_stopping_rounds = 10, nrounds = 1000, params = d.params, verbose = verbose)
    xgbCV$best_iteration
    xgb.d = xgboost(data = x.train, label = d.train, nrounds = xgbCV$best_iteration, verbose = verbose, params = d.params)
    
    y.pred[[k]] = y.test - predict(xgb.y, newdata = x.test)
    d.pred[[k]] = d.test - predict(xgb.d, newdata = x.test)
    
    fits[[k]] = lm(y.pred[[k]] ~ 0 + d.pred[[k]])
  }
  hat_beta = mean(sapply(fits, coefficients))
  
  y.preds = unlist(y.pred)
  d.preds = unlist(d.pred)
  
  fit = lm(y.preds ~ 0 + d.preds)
  fit$coefficients = hat_beta
  return(fit)
}