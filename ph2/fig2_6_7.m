%% sgld vs other
% test
% 1 0.4 0.2
sgld=[52.112676	52.112676	51.509054	50.905433	49.094567	48.088531	52.917505	51.509054	59.557344	60.160966	58.953722	57.947686	57.344064	60.362173	60.160966	60.160966	60.965795	59.557344	61.971831	62.977867	62.977867	60.965795	61.167002	61.971831	63.380282	64.185111	62.977867	63.983903	63.581489	63.782696
];
%1218
lasso=[43.86317907	52.51509054	51.30784708	52.51509054	59.55734406	54.72837022	58.95372233	56.53923541	58.75251509	56.13682093	58.95372233	56.9416499	59.15492958	57.74647887	59.15492958	57.34406439	57.34406439	57.34406439	57.74647887	58.95372233	58.95372233	57.34406439	61.5694165	57.34406439	61.16700201	60.16096579	61.16700201	57.54527163	64.58752515	62.57545272
];
%1218
rf=[53.722334	54.32595573	60.16096579	59.95975855	60.76458753	60.56338028	59.75855131	61.16700201	60.56338028	57.94768612	57.14285714	57.14285714	57.94768612	57.54527163	57.34406439	55.93561368	56.33802817	54.92957746	55.13078471	55.13078471	56.9416499	56.33802817	57.54527163	56.33802817	52.71629779	52.51509054	52.51509054	52.51509054	52.51509054	52.51509054
];
%1218
net=[51.91146881	51.91146881	47.48490946	44.86921529	59.75855131	57.14285714	57.34406439	59.35613682	60.36217304	58.75251509	58.55130785	59.15492958	57.94768612	57.74647887	58.55130785	56.53923541	57.34406439	57.34406439	57.74647887	58.75251509	58.55130785	58.75251509	57.14285714	57.94768612	59.15492958	60.76458753	60.16096579	60.96579477	62.37424547	63.78269618
];
ridge=[51.91146881	51.71026157	51.71026157	52.3138833	52.3138833	50.10060362	55.5331992	54.52716298	55.93561368	54.72837022	53.31991952	54.32595573	57.14285714	55.33199195	54.92957746	57.14285714	56.13682093	58.14889336	57.14285714	60.36217304	59.95975855	59.15492958	58.75251509	60.56338028	59.75855131	61.36820926	60.16096579	60.56338028	60.16096579	60.56338028
];
pca=[51.50905433	51.91146881	53.722334	48.28973843	56.74044266	48.28973843	52.71629779	50.10060362	53.92354125	53.722334	45.87525151	54.92957746	50.50301811	56.13682093	54.72837022	56.53923541	52.3138833	56.53923541	55.93561368	54.72837022	56.53923541	55.73440644	56.13682093	52.3138833	56.9416499	56.33802817	58.55130785	56.9416499	56.53923541	55.33199195
];

figure
subplot(1,2,2)
sgld_l=plot(sgld,'r');
hold on
lasso_l=plot(lasso);
hold on
rf_l=plot(rf);
hold on
net_l=plot(net);
hold on
ridge_l=plot(ridge);
hold on
pca_l=plot(pca);
grid on
legend([sgld_l,lasso_l,net_l,rf_l,pca_l,ridge_l],'SGLD','LASSO','Elastic Net','Random Forest','PCA','Ridge');

title('ROSMAP Test');
ylabel('Accuracy(%)');
xlabel('Feature Number');
set(gca,'FontName','Times New Roman','FontSize',20,'FontWeight','Bold');

%%
% train
% 1 0.4 0.2
sgld=[54.918033	55.122951	55.327869	55.327869	56.147541	56.147541	55.737705	54.918033	66.598361	67.008197	66.598361	66.598361	66.188525	65.778689	65.778689	65.163934	65.57377	65.57377	65.57377	66.188525	67.622951	67.622951	69.672131	69.057377	69.877049	69.262295	70.081967	69.467213	70.081967	70.081967
];
%1218
lasso=[55.73770492	54.71311475	60.86065574	61.8852459	65.16393443	65.57377049	65.98360656	64.3442623	65.57377049	64.54918033	66.18852459	65.98360656	65.98360656	65.16393443	65.77868852	65.98360656	66.18852459	64.75409836	65.16393443	66.59836066	66.59836066	66.80327869	65.36885246	67.21311475	67.41803279	67.41803279	67.41803279	67.21311475	66.59836066	66.80327869
];
%1218
rf=[66.39344262	67.41803279	71.51639344	71.10655738	70.49180328	71.10655738	70.49180328	73.36065574	72.54098361	69.05737705	73.1557377	71.72131148	73.97540984	72.74590164	74.18032787	71.92622951	73.7704918	72.95081967	72.33606557	72.95081967	71.51639344	72.33606557	63.7295082	61.68032787	60.86065574	58.19672131	57.37704918	55.12295082	53.89344262	53.68852459
];
%1218
net=[54.71311475	54.71311475	62.09016393	61.8852459	65.36885246	64.95901639	64.75409836	64.3442623	64.95901639	65.77868852	65.77868852	65.98360656	65.98360656	65.57377049	65.36885246	67.21311475	65.16393443	65.36885246	65.77868852	65.77868852	65.57377049	66.59836066	66.18852459	65.36885246	64.75409836	67.00819672	66.80327869	67.62295082	67.00819672	67.82786885
];
ridge=[58.19672131	59.01639344	59.42622951	62.5	61.68032787	63.1147541	63.93442623	66.18852459	68.23770492	69.87704918	70.28688525	68.85245902	71.51639344	71.92622951	71.51639344	75.20491803	75.81967213	76.8442623	76.8442623	81.96721311	82.99180328	83.60655738	83.19672131	82.37704918	83.40163934	82.78688525	83.40163934	83.19672131	84.22131148	83.40163934
];
pca=[57.58196721	56.55737705	56.96721311	57.58196721	62.09016393	67.21311475	70.49180328	71.31147541	73.1557377	72.74590164	73.1557377	73.1557377	74.18032787	74.18032787	72.95081967	74.3852459	74.18032787	73.56557377	74.79508197	75	74.3852459	72.74590164	74.79508197	75	74.3852459	74.18032787	74.3852459	75	74.18032787	75.20491803
];
subplot(1,2,1)
sgld_2=plot(sgld,'r');
hold on
lasso_2=plot(lasso);
hold on
rf_2=plot(rf);
hold on
net_2=plot(net);
hold on
ridge_2=plot(ridge);
hold on
pca_2=plot(pca);
grid on
legend([sgld_2,lasso_2,net_2,rf_2,pca_2,ridge_2],'SGLD','LASSO','Elastic Net','Random Forest','PCA','Ridge');

title('GEO Train');
ylabel('Accuracy(%)');
xlabel('Feature Number');
set(gca,'FontName','Times New Roman','FontSize',20,'FontWeight','Bold');

%% loc
%1218
sgld_loc=[554191	1212037	1320355	1321949	1322197	1324869	1438043	1438491	1445237	1447653	1448689	1454588	2188739	2195073	2200195	2266605	2278159	2285604	3087097	3092610	3097495	3349810	3663082	3668186	3669406	3714560	51342927	78460545	111571814	241700323
];
ridge_loc=[30018734	141694727	87437853	129387915	36397065	1805819	56179041	67010072	129074749	55058052	144720821	74900038	64568351	30673725	23027033	69173193	40716642	24000563	131860025	205885516	49401569	175980091	2769563	8788859	101621138	71986889	237988139	10798653	16790972	1364173
];
%1218
lasso_loc=[66892	88446971	88409881	88360018	88344614	88344454	88328391	88512755	88295459	88174249	88126069	88120588	88088407	88086615	88030594	88218727	88593304	88627650	88651049	89365313	89364656	89301921	89274479	89274369	89231437	89211759	89065432	89061922	88988667	88955359
];
%1218
rf_loc=[926764	144743304	205885516	161061483	155631053	180027591	64568351	142738830	104221930	87025633	42152629	30018734	200584005	131077956	31963845	131229360	47533427	23156926	114196774	69184265	20464767	54654136	129387915	114417780	54582177	47517775	15016776	55085465	90760882	63660959
];
%1218
net_loc=[66892	88446971	88409881	88360018	88344614	88344454	88328391	88512755	88295459	88174249	88126069	88120588	88088407	88086615	88030594	88218727	88593304	88627650	88651049	89365313	89364656	89301921	89274479	89274369	89231437	89211759	89065432	89061922	88988667	88955359
];

figure
subplot(5,1,1)
stem(sgld_loc,ones(size(sgld_loc)),'linewidth',1.5);
xlabel('(a) SGLD');
set(gca,'xlim',[0,2.5e8],'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');

subplot(5,1,2)
stem(rf_loc,ones(size(rf_loc)),'linewidth',1.5);
xlabel('(b) Random Forest');
set(gca,'xlim',[0,2.5e8],'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');

subplot(5,1,3)
stem(ridge_loc,ones(size(ridge_loc)),'linewidth',1.5);
xlabel('(c) Ridge');
set(gca,'xlim',[0,2.5e8],'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');

subplot(5,1,4)
stem(net_loc,ones(size(net_loc)),'linewidth',1.5);
xlabel('(d) Elastic Net');
set(gca,'xlim',[0,2.5e8],'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');

subplot(5,1,5)
stem(lasso_loc,ones(size(lasso_loc)),'linewidth',1.5);
xlabel('(e) LASSO');
set(gca,'xlim',[0,2.5e8],'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');