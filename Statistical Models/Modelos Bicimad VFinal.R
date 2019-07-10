data_set <- read.table('data_final', sep = ',', header = TRUE, fileEncoding = 'UTF8')

test <-data.frame(data_set)
test$X <- test$season <- test$temp_media <- test$weekend <- test$wind_speed <- NULL
test$year <- factor(test$year)
test$month <- factor(test$month)
test$day <- factor(test$day)
test$unicom <- factor(test$unicom)
test$holiday <- factor(test$holiday)
test2 <- data.frame(test)
test2 <- na.omit(test2)
test2$wind_speed <- test2$temp_max2 <- NULL
test2$date <- as.Date(test2$date)

test4 <- test2[, c("anual_total_use_day", "precipitation", "temp_max", "temp_min", "wind_speed",
                   "dioxido_nitrogeno", "min_sun")]
library(corrplot)
res <- cor(test4, method = c("pearson", "kendall", "spearman"))
corrplot(res, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

poi.mod <- glm(anual_total_use_day ~ precipitation + temp_max + temp_max2 + temp_min 
               + dioxido_nitrogeno + day + month + year + min_sun + holiday 
               + unicom, family = poisson, data = test2)
exp(poi.mod$coef)
summary(poi.mod)

c(deviance = poi.mod$deviance, d.f = poi.mod$df.residual)


n_groups <- 10

mascara <- sample(rep(1:n_groups, nrow(test2) / n_groups), nrow(test2), replace = TRUE)

tmp <- lapply(1:n_groups, function(i){
  modelo <- glm(anual_total_use_day ~ precipitation + temp_max + temp_max2 + temp_min 
                + dioxido_nitrogeno + day + month + year + min_sun + holiday 
                + unicom, family = poisson, data = test2[mascara != i,])
  tmp <- test2[mascara == i,]
  tmp$preds <- predict(modelo, tmp, type = "response")
  tmp
})

res <- do.call(rbind, tmp)
rmse_poi <- sqrt(mean((res$anual_total_use_day - res$preds)^2))

tmp <- lapply(1:n_groups, function(i){
  modelo2 <- glm(anual_total_use_day ~ precipitation + temp_max + temp_max2 + temp_min 
                 + dioxido_nitrogeno + day * month * year + min_sun + holiday 
                 + unicom, family = poisson, data = test2[mascara != i,])
  tmp <- test2[mascara == i,]
  tmp$preds <- predict(modelo2, tmp, type = "response")
  tmp
})

res2 <- do.call(rbind, tmp)

rmse_poi2 <- sqrt(mean((res2$anual_total_use_day - res2$preds)^2))
plot(res2$preds, res2$anual_total_use_day, ylim=c(1000,15500), xlim=c(1000,15500), 
     main = 'Real vs. Predicciones Poisson', xlab = 'Usos Predichos', ylab = 'Usos Reales')
abline(a = 0, b = 1, col = "red")


poi.mod2 <- glm(anual_total_use_day ~ precipitation + temp_max + temp_max2 + temp_min 
                + dioxido_nitrogeno + day * month * year + min_sun + holiday 
                + unicom, family = poisson, data = test2)

exp(poi.mod2$coef)
summary(poi.mod2)


c(deviance = poi.mod2$deviance, d.f = poi.mod2$df.residual)


library(MASS)

nb.mod <- glm.nb(anual_total_use_day ~ precipitation + temp_max + temp_max2 + temp_min 
                 + dioxido_nitrogeno + day + month + year + min_sun + holiday 
                 + unicom, link = log, data = test2)

exp(nb.mod$coef)
summary(nb.mod)

c(theta = summary(nb.mod)$theta, deviance = nb.mod$deviance, d.f = nb.mod$df.residual)


tmp <- lapply(1:n_groups, function(i){
  modelo3 <- glm.nb(anual_total_use_day ~ precipitation + temp_max + temp_max2 + day * month * 
                   year + min_sun + holiday + unicom, link = log, data = test2[mascara != i,])
  tmp <- test2[mascara == i,]
  tmp$preds <- predict(modelo3, tmp, type = "response")
  tmp
})

resbn <- do.call(rbind, tmp)

rmse_bn <- sqrt(mean((resbn$anual_total_use_day - resbn$preds)^2))

plot(resbn$preds, resbn$anual_total_use_day, ylim=c(1000,15500), xlim=c(1000,15500), 
     main = 'Real vs. Predicciones Poisson', xlab = 'Usos Predichos', ylab = 'Usos Reales')
abline(a = 0, b = 1, col = "red")


nb.mod2 <- glm.nb(anual_total_use_day ~ precipitation + temp_max + temp_max2 + day + month 
                  + year + min_sun + holiday + unicom, link = log, data = test2)

library(mvinfluence)
influencePlot(poi.mod2, ylim=c(-60,90), main = 'Variación Residual Poisson')
influencePlot(nb.mod, ylim=c(-60,90), main = 'Variación Residual Binomial Negativa')

anova(nb.mod, nb.mod2)
anova(nb.mod2, poi.mod)

1 - pchisq(summary(nb.mod2)$deviance, summary(nb.mod2)$df.residual)
1 - pchisq(summary(poi.mod)$deviance, summary(poi.mod)$df.residual)

library(caret)
library(randomForest)
library(mlbench)
library(e1071)
library(dplyr)

customRF <- list(type = "Regression",
                 library = "randomForest",
                 loop = NULL)

customRF$parameters <- data.frame(parameter = c("mtry", "ntree"),
                                  class = rep("numeric", 2),
                                  label = c("mtry", "ntree"))

customRF$grid <- function(x, y, len = NULL, search = "grid") {}

customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs) {
  randomForest(x, y,
               mtry = param$mtry,
               ntree=param$ntree)
}

customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)

customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")

customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes


tc_RF <- trainControl(method = "cv", number = 5)
tg_RF = expand.grid(.mtry=c(6:16),.ntree=c(100, 500, 2000, 5000))

custom <- train(anual_total_use_day ~ .,
                data = test2,
                method = customRF,
                metric = "RMSE", 
                tuneGrid=tg_RF,  
                trControl=tc_RF)

summary(custom)
plot(custom)
custom$bestTune


tg_RF2 = expand.grid(.mtry=c(6:13),.ntree=c(100, 500, 2000, 5000))

custom2 <- train(anual_total_use_day ~ .,
                data = test3,
                method = customRF,
                metric = "RMSE", 
                tuneGrid=tg_RF2,  
                trControl=tc_RF)

summary(custom2)
plot(custom2)
custom2$bestTune

rf <- randomForest(anual_total_use_day ~ ., mtry = 11, ntree = 5000, importance = TRUE, data = test2)
varImpPlot(rf)

tmp <- lapply(1:n_groups, function(i){
  modelo3 <- randomForest(anual_total_use_day ~ ., mtry = 16, ntree = 5000, data = test2[mascara != i,])
  tmp <- test2[mascara == i,]
  tmp$preds <- predict(modelo3, tmp)
  tmp
  })

resrf <- do.call(rbind, tmp)
rmse_rf <- sqrt(mean((resrf$anual_total_use_day - resrf$preds)^2))

plot(resrf$preds, resrf$anual_total_use_day, ylim=c(1000,15500), xlim=c(1000,15500), 
     main = 'Real vs. Predicciones Random Forest', xlab = 'Usos Predichos', ylab = 'Usos Reales')
abline(a = 0, b = 1, col = "red")


library(DALEX)
library(ceterisParibus)


tst1 <- tst[, c("anual_total_use_day", "precipitation", "temp_max", "temp_max2", "temp_min", "dioxido_nitrogeno", "day",
                "month", "year", "holiday", "unicom", "min_sun")]

poi.mod <- glm(anual_total_use_day ~ precipitation + temp_max + temp_max2 + temp_min 
                + dioxido_nitrogeno + day + month + year + min_sun + holiday 
                + unicom, family = poisson, data = tst1)

tst2 <- tst[, c("anual_total_use_day", "precipitation", "temp_max", "temp_max2", "day",
                "month", "year", "holiday", "unicom", "min_sun")]

nb.mod <- glm.nb(anual_total_use_day ~ precipitation + temp_max + temp_max2 + day + month 
                  + year + min_sun + holiday + unicom, link = log, data = tst2)

tstrf <- tst[, c("anual_total_use_day", "date", "precipitation", "temp_max", "temp_min", "dioxido_nitrogeno", "day",
                "month", "year", "holiday", "unicom", "min_sun", "wind_speed", "weekend")]

rf <- randomForest(anual_total_use_day ~ ., mtry = 14, ntree = 5000, importance = TRUE, data = tst)

explainer_poi<- explain(poi.mod, data = tst1[,2:12], y = tst1$anual_total_use_day)
explainer_nb<- explain(nb.mod, data = tst2[,2:10], y = tst2$anual_total_use_day)
explainer_rf<- explain(rf, data = tst[,2:14], y = tst$anual_total_use_day)

explainer_rf2<- explain(rf, data = tstrf[,2:14], y = tstrf$anual_total_use_day)

new <- tst[400, ]
new1 <- tst1[400, ]
new2 <- tst2[400, ]

wi_poi <- what_if(explainer_poi, observation = new1)
wi_nb <- what_if(explainer_nb, observation = new2)
wi_rf <- what_if(explainer_rf, observation = new)
wi_rf2 <- what_if(explainer_rf2, observation = new)

plot(wi_nb, ylim=c(0,18000))
plot(wi_poi)
plot(wi_rf2)
plot(wi_nb, wi_poi)

pc <- read.table('plz_cebada', sep = ',', header = TRUE, fileEncoding = 'UTF8')
pc2 <-data.frame(pc)
pc2$anual_total_use_day <- pc2$season <- pc2$weekend <- pc2$X <- pc2$year <- NULL
pc2$year <- factor(pc2$year)
pc2$month <- factor(pc2$month)
pc2$day <- factor(pc2$day)
pc2$unicom <- factor(pc2$unicom)
pc2$holiday <- factor(pc2$holiday)

pc2$plaza_cebada <- as.integer(pc2$plaza_cebada)
pc2$plaza_cebada <- as.integer(pc2$plaza_cebada)

poi.pc <- glm(plaza_cebada ~ precipitation + temp_max + temp_max2 + temp_min 
               + dioxido_nitrogeno + day + month + year + min_sun + holiday 
               + unicom, family = poisson, data = pc2)

exp(poi.pc$coef)
summary(poi.pc)

n_groups <- 10

mascara <- sample(rep(1:n_groups, nrow(pc2) / n_groups), nrow(pc2), replace = TRUE)

tmp <- lapply(1:n_groups, function(i){
  modelo <- glm(plaza_cebada ~ precipitation + temp_max + temp_max2 
                + dioxido_nitrogeno + month * year + min_sun + day + holiday 
                + unicom, family = poisson, data = pc2[mascara != i,])
  tmp <- pc2[mascara == i,]
  tmp$preds <- predict(modelo, tmp, type = "response")
  tmp
})

res <- do.call(rbind, tmp)
rmse_pc <- sqrt(mean((res$plaza_cebada - res$preds)^2))

plot(res$preds, res$plaza_cebada, ylim=c(0,200), xlim=c(0,200),
     main = 'Real vs. Predicciones Poisson', xlab = 'Usos Predichos', ylab = 'Usos Reales')
abline(a = 0, b = 1, col = "red")

nb.pc <- glm.nb(plaza_cebada ~ precipitation + temp_max + temp_max2 + day + month 
                  + min_sun + holiday + unicom, link = log, data = pc2)
exp(nb.pc$coef)
plotsummary(nb.pc)

pc2$date <- as.Date(pc2$date)

tc_RF <- trainControl(method = "cv", number = 10)
tg_RF = expand.grid(.mtry=c(6:13))

custom <- train(plaza_cebada ~ .,
                data = pc2,
                method = 'rf',
                metric = "RMSE", 
                tuneGrid=tg_RF,  
                trControl=tc_RF)
plot(custom)

pc2$temp_max2 <- NULL

rf.pc <- randomForest(plaza_cebada ~., mtry = 11, ntree = 5000, importance = TRUE, data = pc2)
varImpPlot(rf.pc)

tmp <- lapply(1:n_groups, function(i){
  modelo <- randomForest(plaza_cebada ~., mtry = 11, ntree = 5000, importance = TRUE, data = pc2[mascara != i,])
  tmp <- pc2[mascara == i,]
  tmp$preds <- predict(modelo, tmp)
  tmp
})

res_rf <- do.call(rbind, tmp)
rmse_rf <- sqrt(mean((res_rf$plaza_cebada - res_rf$preds)^2))
plot(res_rf$preds, res_rf$plaza_cebada, ylim=c(0,225), xlim=c(0,225),
     main = 'Real vs. Predicciones RF', xlab = 'Usos Predichos', ylab = 'Usos Reales')
abline(a = 0, b = 1, col = "red")

pc2$temp_max2 <- NULL

rf2.pc <- randomForest(plaza_cebada ~., mtry = 12, ntree = 5000, importance = TRUE, data = pc2)
varImpPlot(rf2.pc)

tmp <- lapply(1:n_groups, function(i){
  modelo <- randomForest(plaza_cebada ~., mtry = 12, ntree = 5000, importance = TRUE, data = pc2[mascara != i,])
  tmp <- pc2[mascara == i,]
  tmp$preds <- predict(modelo, tmp)
  tmp
})

res_rf2 <- do.call(rbind, tmp)
rmse_rf <- sqrt(mean((res_rf2$plaza_cebada - res_rf2$preds)^2))

stations <- read.table('stations', sep = ',', header = TRUE, fileEncoding = 'UTF8')

pm <-data.frame(stations)
pm$anual_total_use_day <- pm$season <- pm$weekend <- pm$X <- pm$temp_media <- pm$year <- pm$matadero <- NULL
pm$month <- factor(pm$month)
pm$day <- factor(pm$day)
pm$unicom <- factor(pm$unicom)
pm$holiday <- factor(pm$holiday)
which(is.na(pm$paseo_moret))
pc2$paseo_moret <- pm$paseo_moret
pm2 <-data.frame(pc2)
pm2$plaza_cebada <- NULL
pm2 <- na.omit(pm2)
pm2$temp_max2 <- pm2$temp_max*pm2$temp_max

nb.pm <- glm.nb(paseo_moret ~ precipitation + temp_max + temp_max2 + day + month 
                + min_sun + holiday + unicom, link = log, data = pm2)
exp(nb.pm$coef)
summary(nb.pm)

mascara_pm <- sample(rep(1:n_groups, nrow(pm2) / n_groups), nrow(pm2), replace = TRUE)

tmp <- lapply(1:n_groups, function(i){
  modelo <- glm(paseo_moret ~ precipitation + temp_max + temp_max2 
                + dioxido_nitrogeno + day + month + min_sun +  + holiday 
                + unicom, family = poisson, data = pm2[mascara_pm != i,])
  tmp <- pm2[mascara_pm == i,]
  tmp$preds <- predict(modelo, tmp, type = "response")
  tmp
})

res <- do.call(rbind, tmp)
rmse_pm <- sqrt(mean((res$paseo_moret - res$preds)^2))
plot(res$preds, res$paseo_moret, ylim=c(0,37), xlim=c(0,37),
     xlab = 'Usos Predichos', ylab = 'Usos Reales')
abline(a = 0, b = 1, col = "red")

pm2$temp_max2 <- NULL

rf.pm <- randomForest(paseo_moret ~., mtry = 11, ntree = 5000, importance = TRUE, data = pm2)
varImpPlot(rf.pm)

tmp <- lapply(1:n_groups, function(i){
  modelo <- randomForest(paseo_moret ~., mtry = 11, ntree = 5000, importance = TRUE, data = pm2[mascara_pm != i,])
  tmp <- pm2[mascara_pm == i,]
  tmp$preds <- predict(modelo, tmp)
  tmp
})

res_rfpm <- do.call(rbind, tmp)
rmse_pm <- sqrt(mean((res_rfpm$paseo_moret - res_rfpm$preds)^2))
plot(res_rfpm$preds, res_rfpm$paseo_moret, ylim=c(0,37), xlim=c(0,37),
     xlab = 'Usos Predichos', ylab = 'Usos Reales')
abline(a = 0, b = 1, col = "red")

stations <- read.table('stations', sep = ',', header = TRUE, fileEncoding = 'UTF8')

mata <-data.frame(stations)
pc2$matadero <- mata$matadero
mata <-data.frame(pc2)
mata$plaza_cebada <- mata$paseo_moret <- NULL
mata2 <- na.omit(mata)
mata2$temp_max2 <- mata2$temp_max*mata2$temp_max

nb.mat <- glm.nb(matadero ~ precipitation + temp_max + temp_max2 + day + month 
                + min_sun + holiday + unicom, link = log, data = mata2)

exp(nb.mat$coef)
summary(nb.mat)

mascara_mat <- sample(rep(1:n_groups, nrow(mata2) / n_groups), nrow(mata2), replace = TRUE)

tmp <- lapply(1:n_groups, function(i){
  modelo <- glm(matadero ~ precipitation + temp_max + temp_max2 
                + dioxido_nitrogeno + day + month + min_sun + holiday 
                + unicom, family = poisson, data = mata2[mascara_mat != i,])
  tmp <- mata2[mascara_mat == i,]
  tmp$preds <- predict(modelo, tmp, type = "response")
  tmp
})

res <- do.call(rbind, tmp)
rmse_mata <- sqrt(mean((res$matadero - res$preds)^2))
plot(res$preds, res$matadero, ylim=c(0,210), xlim=c(0,210),
     xlab = 'Usos Predichos', ylab = 'Usos Reales')
abline(a = 0, b = 1, col = "red")

mata2$temp_max2 <- NULL

rf.mat <- randomForest(matadero ~., mtry = 11, ntree = 5000, importance = TRUE, data = mata2)
varImpPlot(rf.mat)

tmp <- lapply(1:n_groups, function(i){
  modelo <- randomForest(matadero ~., mtry = 11, ntree = 5000, importance = TRUE, data = mata2[mascara_mat != i,])
  tmp <- mata2[mascara_mat == i,]
  tmp$preds <- predict(modelo, tmp)
  tmp
})

res_mat <- do.call(rbind, tmp)
rmse_mat <- sqrt(mean((res_mat$matadero - res_mat$preds)^2))
plot(res_mat$preds, res_mat$matadero, ylim=c(0,210), xlim=c(0,210),
     xlab = 'Usos Predichos', ylab = 'Usos Reales')
abline(a = 0, b = 1, col = "red")


