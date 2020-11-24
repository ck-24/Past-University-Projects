attach(who)
#plot(expenditure,life)
who
lmwho<-lm(life~expenditure, who)
abline(lmwho)
other_countries<-who[-(30),]
other_countries
plot(other_countries$expenditure,other_countries$life)
uk<-who[30,]
xvalues<-who$expenditure
predictions.PI<-data.frame(predict.lm(lmwho,xvalues,interval="prediction",level=0.95))
m<-data.frame(country="Other OECD Countries",life=mean(other_countries$life),tobacco=mean(other_countries$tobacco),expenditure=mean(other_countries$expenditure),alcohol=mean(other_countries$alcohol),obesity=mean(other_countries$obesity))
a<-rbind(who$United,m)
a
uk
cor(life,log(tobacco))
cor(life,log(expenditure))
cor(life,alcohol)
cor(life,log(obesity))
lm1<-lm(life~tobacco+expenditure+alcohol+obesity, who)
lm1
evals<-stdres(lm1)
plot(fitted(lm1),evals)
par(mfrow=c(1,2)) 
hist(evals) 
qqnorm(evals) 
abline(0,1)
ks.test(evals,pnorm,0,1)
lm2<-lm(life~tobacco+expenditure+alcohol, who)
lm3<-lm1<-lm(life~expenditure+alcohol+obesity, who)
lm4<-lm(life~expenditure+obesity, who)
lm5<-lm(life~alcohol+obesity, who)
lm6<-lm(life~tobacco+expenditure+obesity, who)
lm7<-lm(life~tobacco+alcohol+obesity, who)
anova(lm7,lm1)
lm3$coefficients
summary(lm3)
lm8<-lm(life~expenditure,who)
xvalues<-data.frame(expenditure=seq(from=890,to=7540,length=100)) 
predictions99<-data.frame(predict.lm(lm8,xvalues,interval="prediction",level=0.99)) 
predictions95<-data.frame(predict.lm(lm8,xvalues,interval="prediction",level=0.95)) 
plot(mort$expenditure,mort$life) 
lines(xvalues$expenditure,predictions95$fit)  
lines(xvalues$expenditure,predictions95$lwr,lty=4) 
lines(xvalues$expenditure,predictions95$upr,lty=4) 
lines(xvalues$expenditure,predictions99$lwr,lty=3) 
lines(xvalues$expenditure,predictions99$upr,lty=3)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
