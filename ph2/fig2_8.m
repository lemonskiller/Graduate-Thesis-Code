%% 450k
% 450k
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

% 850k
sgld_loc2=[616255	891847	892074	894243	897008	1255217	4222306	4449308	7700550	25190326	43534354	50612978	51666902	55269631	62362877	63143246	77525647	109619019	120982749	138231664	144091190	144091202	145075641	157337316	160600841	166990043	166990418	166990606	234083006	239865707
];
rf_loc2=[55899237	166073114	62817558	63143246	61101399	35805788	227738168	52647383	63095952	83928555	52688698	49760516	50255090	151285192	71065665	102340305	7687741	157167170	28237338	56105843	22281613	30962165	120982749	41107162	62394464	27864174	144091202	129426651	89597131	45113540
];
ridge_loc2=[8304214	157167170	11774062	57685151	52221941	61112438	52690582	74897975	136564118	98091699	26059369	11590303	43620950	144801705	4825493	166073114	29974959	158975614	75544772	4222306	58948089	7726435	30313577	146814450	178919231	97788546	55899237	64339702	45304125	32835234
];
net_loc2=[318170	89510812	89597131	89669546	89730460	89778966	90104326	90181153	90198946	89359101	90327871	90477973	90507955	90601762	90601835	90646958	90902325	90930366	90942535	90413840	89296039	89282211	89106767	87716884	87730474	87740256	87839763	87959221	88062718	88114161
];
lasso_loc2=[318170	89510812	89597131	89669546	89730460	89778966	90104326	90181153	90198946	89359101	90327871	90477973	90507955	90601762	90601835	90646958	90902325	90930366	90942535	90413840	89296039	89282211	89106767	87716884	87730474	87740256	87839763	87959221	88062718	88114161
];

figure
subplot(5,1,1)
% 450k
h1=stem(sgld_loc,1.5*ones(size(sgld_loc)),'linewidth',1.5);
hold on
% 850k
h2=stem(sgld_loc2,ones(size(sgld_loc2)),'linewidth',1.5);
xlabel('(a) SGLD');
legend([h1,h2],'450k','850k');
set(gca,'xlim',[0,3.5e8],'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');

subplot(5,1,2)
% 450k
h1=stem(rf_loc,1.5*ones(size(rf_loc)),'linewidth',1.5);
hold on
% 850k
h2=stem(rf_loc2,ones(size(rf_loc2)),'linewidth',1.5);
xlabel('(b) Random Forest');
legend([h1,h2],'450k','850k');
set(gca,'xlim',[0,3.5e8],'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');

subplot(5,1,3)
% 450k
h1=stem(ridge_loc,1.5*ones(size(ridge_loc)),'linewidth',1.5);
hold on
% 850k
h2=stem(ridge_loc2,ones(size(ridge_loc2)),'linewidth',1.5);
xlabel('(c) Ridge');
legend([h1,h2],'450k','850k');
set(gca,'xlim',[0,3.5e8],'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');

subplot(5,1,4)
% 450k
h1=stem(net_loc,1.5*ones(size(net_loc)),'linewidth',1.5);
hold on
% 850k
h2=stem(net_loc2,ones(size(net_loc2)),'linewidth',1.5);
xlabel('(d) Elastic Net');
legend([h1,h2],'450k','850k');
set(gca,'xlim',[0,3.5e8],'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');

subplot(5,1,5)
% 450k
h1=stem(lasso_loc,1.5*ones(size(lasso_loc)),'linewidth',1.5);
hold on
% 850k
h2=stem(lasso_loc2,ones(size(lasso_loc2)),'linewidth',1.5);
xlabel('(e) LASSO');
legend([h1,h2],'450k','850k');
set(gca,'xlim',[0,3.5e8],'YTick',[],'ylim',[0,2],'FontName','Times New Roman','FontSize',12,'FontWeight','Bold');


