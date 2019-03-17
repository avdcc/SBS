# Trabalho de SBS (sistema de recomendação)

Temas : Comercio eletronico 
        Hotelaria e Restauracao

### Artigos e Trabalhos relacionados:
1. https://estudogeral.sib.uc.pt/bitstream/10316/35644/1/Machine%20Learning%20in%20a%20Recommendation%20System.pdf

2. http://www.lbd.dcc.ufmg.br/colecoes/sbsi/2015/029.pdf

3. ftp://ftp.inf.puc-rio.br/pub/docs/techreports/10_19_carvalho.pdf

### Outros Links:
1. https://pt.slideshare.net/EdjalmaQueirozdaSilva/sistemas-recomendacao



### Topicos do Trabalho: 
1. Definições e História
2. Paradigmas de sistemas de recomendação
3. Características de cada paradigma
4. Vantagens para o promotor da recomendação
5. Vantagens para o alvo da recomendação
6. Técnicas de machine learning utilizadas
7. Exemplos característicos


## Definições e História

Bibliografia:-----------------------------------
Indicam-se as referências históricas de cada tema:

    Dietmar Jannach, Markus Zanker, Alexander Felfernig, Gerhard Friedrich, “Recommender
    Systems: An Introduction 1st Edition”, Cambridge University Press, 2011
 
    Francesco Ricci, Lior Rokach, Bracha Shapira, “Recommender Systems Handbook 2nd ed.”,
    Springer, 2015

    Charu C. Aggarwal, “Recommender Systems: The Textbook 1st ed.”, Springer, 2016
---------------------------------------------------    

#### o que é um sistema de recomendação:
##### 1)
    Sistema de Recomendação é um conjunto de algoritmos 
    que utilizam técnicas de Aprendizagem de Máquina (AM)
    e Recuperação da Informação (RI) para gerar recomendações 
    baseadas em algum tipo de filtragem, as mais comuns 
    são: colaborativa (considera a experiência de todos os usuários),
    baseada em conteúdo (considera a experiência do usuário alvo)
    e híbrida (as duas abordagens são consideradas). 

    fonte: http://igti.com.br/blog/como-funcionam-os-sistemas-de-recomendacao/

##### 2)
    Um Sistema de Recomendação combina várias técnicas computacionais para selecionar
    itens personalizados com base nos interesses dos usuários e conforme o contexto no
    qual estão inseridos.[1] Tais itens podem assumir formas bem variadas como, por
    exemplo, livros, filmes, notícias, música, vídeos, anúncios, links patrocinados,
    páginas de internet, produtos de uma loja virtual, etc. Empresas como Amazon,
    Netflix e Google são reconhecidas pelo uso intensivo de sistemas de recomendação
    com os quais obtém grande vantagem competitiva. Empreendimentos brasileiros também
    estão aderindo tecnologias que utilizam um sistema de recomendação, muitas vezes
    com Machine Learning, Deep Learning ou Inteligência Artificial

    fonte: https://pt.wikipedia.org/wiki/Sistema_de_recomenda%C3%A7%C3%A3o

##### 3)
    Os sistemas de recomendação são uma sub-area de aprendizagem de máquina
    (machine learning) e tem por objetivo sugerir itens a um usuário, com base
    em seu histórico de preferências. Podem ser recomendados itens diversos
    como livros, investimentos ou viagens. É amplamente utilizado como uma
    estratégia de marketing, já que ao recomendar produtos que estejam alinhados
    ao interesse do usuário, é mais provável que ele venha adquirir tal produto.
    É possível fazer recomendações comparando as preferências de um usuário com
    um grupo de outros usuários. Também é possível fazer recomendações procurando
    itens com características similares aos que o usuário já demonstrou interesse
    no passado. As preferências do usuário podem ser colhidas implicitamente ou
    explicitamente. Na forma implícita, informações são obtidas através de opções
    de compras passadas, histórico de sites visitados, links clicados, cookies do
    browser ou até mesmo localidade geográfica. Há também a forma explícita de
    averiguar preferências, utilizando feedbacks efetivos, como por exemplo notas
    dadas a um determinado item.

    fonte: https://www.ibm.com/developerworks/br/local/data/sistemas_recomendacao/index.html


#### Contextualização:
##### 1)
    Em resposta à dificuldade das pessoas em escolher entre uma grande variedade
    de produtos e serviços e entre as várias alternativas que lhe são apresentadas,
    surgem os sistemas de recomendação computacionais. A evolução destes sistemas
    e o fato deles trabalharem com grandes bases de informações permitiram que
    recomendações emergentes (não triviais) pudessem ser alcançadas, proporcionando
    ainda maior credibilidade que uma recomendação humana.[2]

    Os proponentes de um dos primeiros sistemas de recomendação, denominado Tapestry,
    desenvolvido no início dos anos 90, criaram a expressão “Filtragem Colaborativa”
    visando designar um tipo de sistema específico no qual a filtragem da informação
    era realizada com o auxílio humano, ou seja, através da colaboração entre os
    grupos interessados.[3]Vários pesquisadores acabaram adotando esta terminologia
    para denominar qualquer tipo de sistema de recomendação subseqüente. Resnick, no
    seu artigo, defendeu o termo “sistemas de recomendação” como terminologia mais
    genérica do que filtragem colaborativa, já que sistemas de recomendação podem
    existir sem nenhuma colaboração entre as pessoas. [4] 


    fonte: https://pt.wikipedia.org/wiki/Sistema_de_recomenda%C3%A7%C3%A3o


##### 2)
    Sistemas de Recomendação podem ser vistos ao realizar buscas em sites de pesquisa da internet, em compras online, ou até mesmo ao visualizamos nossos emails. São o mecanismo por trás da propaganda personalizada que
       recebemos na web, com indicações de sites para visitarmos ou produtos
    para compramos.
    Com o advento do consumo em dispositivos móveis e a propagação o e-commerce,
    sistemas de recomendação tornaram-se um tema extremamente atrativo.
    Através de algoritmos simples e facilmente integráveis a aplicações web,
    eles agregam valor ao negócio online, promovendo itens de consumo direcionados
    a um público alvo.

    Por trás da singela propaganda, estes sistemas utilizam abstrações matemáticas
    de dados. Neste artigo, veremos que eles consistem basicamente em algoritmos
    de filtragem e inferência de dados, que recomendam produtos de acordo com os
    interesses dos usuários. 

    fonte: https://www.ibm.com/developerworks/br/local/data/sistemas_recomendacao/index.html


#### Em quais aplicações podem ser utilizadas:
    É importante avaliar para quais aplicações esses sistemas são viáveis.
    Primeiramente, estas devem obrigatoriamente basear-se em itens sendo
    expostos ou oferecidos a usuários. Em outro caso, o algoritmo perde
    seu sentido.

    Outro ponto importante é que esse mecanismo só é aplicável quando há
    grande volume de dados envolvidos. Isso é necessário para garantir
    que a metodologia seja eficiente, já que, são feitas abstrações
    matemáticas e quanto mais dados, mais apurada a função de abstração,
    e portanto, mais correto o resultado. 

     fonte: https://www.ibm.com/developerworks/br/local/data/sistemas_recomendacao/index.html

## Paradigmas de sistemas de recomendação

#### tipos
1. Sistemas colaborativos
2. Sistemas de Conteúdo
3. Sistemas sociais

fonte: https://www.linkedin.com/pulse/machine-learning-para-sistemas-de-recomenda%C3%A7%C3%A3o-marco-garcia

#### Sistemas colaborativos
##### 1)
    Esses é o paradigma mais famoso. Tão famoso que muitas
    vezes ele é usado como sinônimo de sistemas de recomendação.
    A ideia principal é que dada uma matriz de preferências de
    usuários por produtos, podemos preencher os buracos das preferências
    que ainda não foram coletadas e recomendar os produtos com maior
    taxa de preferência. Uma das grandes vantagens desse sistema é que
    existe uma quantidade enorme de pesquisa feita em cima dele, tornando
    seu comportamento algo bem conhecido. Por conta disso, existem dezenas
    de frameworks e tutoriais que fazem a implementação de um sistema
    desses algo bem simples. Outra grande vantagem é que não precisamos
    das características dos produtos para fazer as recomendações.
    Tudo que você precisa é da ID do usuário e alguma noção de preferência
    dele sobre produtos(nota, quantidade comprada, se comprou/visualizou, etc).
    Isso é uma propriedade bastante útil quando lidamos com produtos abstratos,
    que não têm definições concretas.

    A maior limitação desses sistemas é que eles dependem muito das preferências
    dos usuários para recomendar coisas. Em um cenário de início frio(cold start),
    onde não temos muitos usuários ou o usuário é novo ou o produto é novo, não
    conseguimos gerar recomendaçõe úteis. Por conta disso, podemos entender que 
    sistemas colaborativos têm seu desempenho inversamente proporcional à esparsidade
    das matrizes de utilidade que definem as preferências dos usuários. É por isso que,
    na minha experiência, sistemas colaborativos sempre exigem modificações
    particulares ao problema sendo resolvido, sejam elas nos próprios algoritmos ou
    até nos dados em si.Esses é o paradigma mais famoso. Tão famoso que muitas vezes
    ele é usado como sinônimo de sistemas de recomendação. A ideia principal é que
    dada uma matriz de preferências de usuários por produtos, podemos preencher os
    buracos das preferências que ainda não foram coletadas e recomendar os produtos
    com maior taxa de preferência. Uma das grandes vantagens desse sistema é que 
    existe uma quantidade enorme de pesquisa feita em cima dele, tornando seu
    comportamento algo bem conhecido. Por conta disso, existem dezenas de frameworks e
    tutoriais que fazem a implementação de um sistema desses algo bem simples. Outra
    grande vantagem é que não precisamos das características dos produtos para fazer
    as recomendações. Tudo que você precisa é da ID do usuário e alguma noção de
    preferência dele sobre produtos(nota, quantidade comprada, se 
    comprou/visualizou, etc). Isso é uma propriedade bastante útil quando lidamos
    com produtos abstratos,que não têm definições concretas.

    A maior limitação desses sistemas é que eles dependem muito das preferências dos
    usuários para recomendar coisas. Em um cenário de início frio(cold start), onde
    não temos muitos usuários ou o usuário é novo ou o produto é novo, não conseguimos
    gerar recomendações úteis. Por conta disso, podemos entender que sistemas
    colaborativos têm seu desempenho inversamente proporcional à esparsidade das
    matrizes de utilidade que definem as preferências dos usuários. É por isso que, na
    minha experiência, sistemas colaborativos sempre exigem modificações particulares
    ao problema sendo resolvido, sejam elas nos próprios algoritmos ou até nos dados
    em si.    

    fonte: https://www.linkedin.com/pulse/machine-learning-para-sistemas-de-recomenda%C3%A7%C3%A3o-marco-garcia


##### 2)
    Recomendações Colaborativas: 
    (LINK https://pt.wikipedia.org/wiki/Filtragem_colaborativa)

    O usuário receberá recomendações de itens
    que pessoas com gostos similares aos dele preferiram no passado.
    Este método é subdividido em duas catergorias: a primeira chamada de
    memory-based, e a segunda chamada de model-based; 

    fonte: https://pt.wikipedia.org/wiki/Sistema_de_recomenda%C3%A7%C3%A3o

##### 3)
    Filtragem colaborativa consiste na recomendação de itens que pessoas
    com gosto semelhante preferiram no passado. Analisa-se a vizinhança
    do usuário a partir da regra: "Se um usuário gostou de A e de B, um
    outro usuário que gostou de A também pode gostar de B". Esse tipo de
    recomendação apresenta resultados positivos na prática [LINDEN, Greg.
    SMITH, Brent. YORK Jeremy. Amazon.com Recommendations Item-to-Item
    Collaborative Filtering ], e evita o problema de recomendações repetitivas.
    Uma desvantagem é que requer grande número de informações sobre o usuário
    e sua vizinhança para funcionar precisamente.

    fonte: https://www.ibm.com/developerworks/br/local/data/sistemas_recomendacao/index.html

##### 4)
    Recomenda ao usuário itens que pessoas com gosto semelhante preferiram no
    passado. Em um cenário de recomendação de livros, por exemplo, se dois
    usuários avaliarem muito bem o livro “Harry Potter e a Pedra Filosofal”
    e, em seguida, o primeiro deles avaliar muito bem o livro “A Game of
    Thrones”, é provável que o segundo usuário receba este último livro como
    recomendação de leitura.

    Uma vantagem importante nesse tipo de abordagem é que a filtragem
    colaborativa é independente das propriedades específicas do item.
    Tudo o que você precisa para começar é IDs de usuário e item, e
    alguma noção de preferência desses usuários para os itens. Contudo,
    como desvantagem, requer grande número de informações sobre o
    usuário e usuários com perfil similar para funcionar corretamente.

    fonte: http://coral.ufsm.br/pet-si/index.php/sistemas-de-recomendacao-desvendando-uma-parte-da-magica/


##### 5)
    One approach to the design of recommender systems that has wide use is
    collaborative filtering.[32] Collaborative filtering methods are based on
    collecting and analyzing a large amount of information on users’ behaviors,
    activities or preferences and predicting what users will like based on
    their similarity to other users. A key advantage of the collaborative filtering approach is that it does not rely on machine analyzable content
     and therefore it is capable of accurately recommending complex items such
      as movies without requiring an "understanding" of the item itself.
       Many algorithms have been used in measuring user similarity or
        item similarity in recommender systems. For example, the k-nearest
         neighbor (k-NN) approach[33] and the Pearson Correlation as first 
    implemented by Allen.[34]

    Collaborative filtering is based on the assumption that people who agreed
    in the past will agree in the future, and that they will like similar kinds
    of items as they liked in the past.

    When building a model from a user's behavior, a distinction is often made
    between explicit and implicit forms of data collection.

    Examples of explicit data collection include the following:

    Asking a user to rate an item on a sliding scale.
    Asking a user to search.
    Asking a user to rank a collection of items from favorite to least favorite.
    Presenting two items to a user and asking him/her to choose the better one of them.
    Asking a user to create a list of items that he/she likes.

    Examples of implicit data collection include the following:

    Observing the items that a user views in an online store.
    Analyzing item/user viewing times.[35]
    Keeping a record of the items that a user purchases online.
    Obtaining a list of items that a user has listened to or watched on his/her computer.
    Analyzing the user's social network and discovering similar likes and dislikes.

    The recommender system compares the collected data to similar and dissimilar
     data collected from others and calculates a list of recommended items for the
    user. Several commercial and non-commercial examples are listed in the 
    article on collaborative filtering systems.

    One of the most famous examples of collaborative filtering is item-to-item
    collaborative filtering (people who buy x also buy y), an algorithm 
    popularized by Amazon.com's recommender system.[36] Other examples include:

    As previously detailed, Last.fm recommends music based on a comparison of
     the listening habits of similar users, while Readgeek compares books 
     ratings for recommendations.
    Facebook, MySpace, LinkedIn, and other social networks use collaborative
     filtering to recommend new friends, groups, and other social connections
      (by examining the network of connections between a user and their friends)
      .[1] Twitter uses many signals and in-memory computations for 
      recommending to its users whom they should "follow."[6]

    Collaborative filtering approaches often suffer from three problems: cold start, scalability, and sparsity.[37]

    Cold start: These systems often require a large amount of existing data 
    on a user in order to make accurate recommendations.[10][11]
    Scalability: In many of the environments in which these systems make
     recommendations, there are millions of users and products. Thus, a large 
     amount of computation power is often necessary to calculate 
     recommendations.
    Sparsity: The number of items sold on major e-commerce sites is extremely 
    large. The most active users will only have rated a small subset of the 
    overall database. Thus, even the most popular items have very few ratings.

    A particular type of collaborative filtering algorithm uses matrix 
    factorization, a low-rank matrix approximation technique.[38][39][40]

    Collaborative filtering methods are classified as memory-based and model
     based
    collaborative filtering. A well-known example of memory-based approaches is 
    user-based algorithm[41] and that of model-based approaches is 
    Kernel-Mapping 
    Recommender.[42] 

    fonte: https://en.wikipedia.org/wiki/Recommender_system

##### 6)
    These kinds of systems utilize user interactions to filter for items of 
    interest. We can visualize the set of interactions with a matrix, where each 
    entry (i,j)(i, j)(i,j) represents the interaction between user iii and item 
    jjj. An interesting way of looking at collaborative filtering is to think of it 
    as a generalization of classification and regression. While in these cases we 
    aim to predict a variable that directly depends on other variables (features), 
    in collaborative filtering there is no such distinction of feature variables 
    and class variables.

    Visualizing the problem as a matrix, we don’t look to predict the values of a 
    unique column, but rather to predict the value of any given entry.

    In short, collaborative filtering systems are based on the assumption that if a 
    user likes item A and another user likes the same item A as well as another 
    item, item B, the first user could also be interested in the second item. Hence,
     they aim to predict new interactions based on historical ones. There are two 
     types of methods to achieve this goal: memory-based and model-based.

    fonte: https://tryolabs.com/blog/introduction-to-recommender-systems/

#### Sistemas de conteúdo
##### 1)
    Esses recebem uma lista de usuários e suas preferências por itens, recomendando
    itens similares àqueles já comprados pelo usuário, dado uma noção do domínio dos
    produtos. A vantagem desse paradigma é que ele não sofre tanto do cold start visto
    nos sistemas colaborativos. Com uma quantidade pequena de preferências, já é
    possível criar uma vasta quantidade de recomendações úteis (todos que já criaram
    uma conta no Netflix podem se lembrar do processo de criação, onde ele te pedia
    para marcar filmes que você já viu e suas preferências por eles). Essas
    recomendações criadas, então, podem ser injetadas em um sistema colaborativo, por
    exemplo, melhorando ainda mais as recomendações.

    Em muitos casos, sistemas de conteúdo são a abordagem mais natural.
    Por exemplo, quando recomendamos artigos de jornal, é intuitivo querer fazer
    recomendações com base no conteúdo dos artigos. Essa abordagem se estende para
    situações onde temos metadados sobre os itens, como em filmes ou livros.

    Alguns dos problemas com essa abordagem surgem quando a similaridade entre esses
    itens não é claramente definida. Porém, mesmo quando a similaridade é clara, os
    resultados do sistema de conteúdo tendem a ser muito homogêneos. Isso faz com que
    os itens recomendados nunca caiam fora da zona de conforto definida no início do
    registro dos usuários. As pessoas mudam com o tempo, assim como mudam suas
    preferências. Sistemas de conteúdo simples têm dificuldade para acompanhar essas
    mudanças.

    fonte: https://www.linkedin.com/pulse/machine-learning-para-sistemas-de-recomenda%C3%A7%C3%A3o-marco-garcia


##### 2)
    Recomendações Baseadas em Conteúdo: 
    (LINK https://pt.wikipedia.org/wiki/Filtragem_baseada_em_conte%C3%BAdo)

    O usuário receberá recomendações de itens similares a itens preferidos
    no passado;[8]

    fonte: https://pt.wikipedia.org/wiki/Sistema_de_recomenda%C3%A7%C3%A3o

##### 3)
    Um sistema de recomendação baseado em conteúdo recomenda ao usuário
    produtos que sejam semelhantes ao que ele preferiu no passado.
    A recomendação é feita a partir de tags "descritoras" de itens.
    Itens com características próximas destas tags são recomendados.
    Em um cenário de recomendação de filmes, por exemplo, um usuário
    que, assiste e gosta do filme "Matrix" teria recomendações do gênero
    ação e ficção científica.

    Vantagens deste tipo de sistema é que são simples para dados textuais
    e não necessitam de muitas informações sobre um usuário para sugerir
    itens. Todavia, além de serem difíceis de aplicar em contextos multimídia,
    podem oferecer recomendações repetitivas, recomendando sempre assuntos
    que o usuário já conhece.

    fonte: https://www.ibm.com/developerworks/br/local/data/sistemas_recomendacao/index.html

##### 4)
    Recomenda ao usuário produtos que sejam semelhantes ao que ele preferiu
    no passado. Em um cenário de recomendação de filmes, um usuário que
    assiste e gosta do filme “Star Wars” teria, por exemplo, recomendações
    de filmes do gênero fantasia e ficção científica.

    A principal vantagem da recomendação baseada em conteúdo sobre a
    filtragem colaborativa é que ele não requer tanto feedback do
    usuário para começar.  Entretanto, um problema surge quando
    a similaridade de itens não é tão facilmente definida, o que
    pode resultar em recomendações bastante homogêneas e repetitivas.

    fonte: https://www.ibm.com/developerworks/br/local/data/sistemas_recomendacao/index.html


##### 5)
    Another common approach when designing recommender systems is content-based filtering. Content-based filtering methods are based on a description of the 
    item and a profile of the user’s preferences.[43][44]

    In a content-based recommender system, keywords are used to describe the items 
    and a user profile is built to indicate the type of item this user likes. In 
    other words, these algorithms try to recommend items that are similar to those 
    that a user liked in the past (or is examining in the present). In particular, 
    various candidate items are compared with items previously rated by the user 
    and the best-matching items are recommended. This approach has its roots in 
    information retrieval and information filtering research.

    To abstract the features of the items in the system, an item presentation 
    algorithm is applied. A widely used algorithm is the tf–idf representation 
    (also called vector space representation).

    To create a user profile, the system mostly focuses on two types of information:

    1. A model of the user's preference.

    2. A history of the user's interaction with the recommender system.

    Basically, these methods use an item profile (i.e., a set of discrete 
    attributes and features) characterizing the item within the system. The system 
    creates a content-based profile of users based on a weighted vector of item 
    features. The weights denote the importance of each feature to the user and can
     be computed from individually rated content vectors using a variety of 
     techniques. Simple approaches use the average values of the rated item vector 
     while other sophisticated methods use machine learning techniques such as 
     Bayesian Classifiers, cluster analysis, decision trees, and artificial neural
      networks in order to estimate the probability that the user is going to like 
      the item.[45]

    Direct feedback from a user, usually in the form of a like or dislike button,
     can be used to assign higher or lower weights on the importance of certain 
     attributes (using Rocchio classification or other similar techniques).

    A key issue with content-based filtering is whether the system is able to learn
     user preferences from users' actions regarding one content source and use them
      across other content types. When the system is limited to recommending 
      content of the same type as the user is already using, the value from the
       recommendation system is significantly less than when other content types
        from other services can be recommended. For example, recommending news 
        articles based on browsing of news is useful, but would be much more useful
         when music, videos, products, discussions etc. from different services can
          be recommended based on news browsing.

    Pandora Radio is an example of a content-based recommender system that plays 
    music with similar characteristics to that of a song provided by the user as an
     initial seed. There are also a large number of content-based recommender 
     systems aimed at providing movie recommendations, a few such examples include 
     Rotten Tomatoes, Internet Movie Database, Jinni, Rovi Corporation, and Jaman.
      Document related recommender systems aim at providing document 
      recommendations to knowledge workers. Public health professionals have been
       studying recommender systems to personalize health education and 
       preventative strategies.

    fonte: https://en.wikipedia.org/wiki/Recommender_system

##### 6)
    These systems make recommendations using a user’s item and profile features.
     They hypothesize that if a user was interested in an item in the past, they 
     will once again be interested in it in the future. Similar items are usually 
     grouped based on their features. User profiles are constructed using 
     historical interactions or by explicitly asking users about their interests.
      There are other systems, not considered purely content-based, which utilize 
      user personal and social data.

    One issue that arises is making obvious recommendations because of excessive 
    specialization (user A is only interested in categories B, C, and D, and the 
    system is not able to recommend items outside those categories, even though 
    they could be interesting to them).

    Another common problem is that new users lack a defined profile unless they are
     explicitly asked for information. Nevertheless, it is relatively simple to add
      new items to the system. We just need to ensure that we assign them a group
       according to their features.

    fonte: https://tryolabs.com/blog/introduction-to-recommender-systems/

#### Sistemas sociais
##### 1)
    Descobertos com o boom das redes sociais, esses sistemas se aproveitam dos dados
    de comportamento e relacionamentos gerados por redes sociais. Exemplos desses
    sistemas são coisas como itens curtidos por amigos e pessoas demograficamente
    similares. Tais sistemas não necessitam de nenhuma informação de preferência de
    usuários específicos para fazerem recomendações e, na minha experiência, mesmo os
    sistemas mais simples que seguem esse paradigma são capazes de gerar resultados
    desesperadoramente precisos. Por exemplo, só de somar os likes dos amigos próximos
    a uma pessoa nos faz capazes de nos pintar uma descrição bem precisa dos gostos
    dessa pessoa.

    Dado esse poder de sistemas de recomendação sociais, não é surpresa que as
    empresas que controlam esses dados não os liberam com facilidade. Isso significa
    que para o cientista de dados comum, criar um sistema desses é praticamente
    impossível. Porém, mesmo quando esses dados estão disponíveis, é difícil
    utilizá-los sem assustar seus usuários. Privacidade tem se tornado um problema
    sério com essa era das redes sociais e isso limita o quanto podemos explorar nesse
    sentido.

    fonte: https://www.linkedin.com/pulse/machine-learning-para-sistemas-de-recomenda%C3%A7%C3%A3o-marco-garcia


#### Metodos Hibridos
##### 1)
    Métodos Híbridos: Estes métodos combinam tanto estratégias de recomendação
    baseadas em conteúdo quanto estratégias baseadas em colaboração.

    fonte: https://pt.wikipedia.org/wiki/Sistema_de_recomenda%C3%A7%C3%A3o

##### 2)
    Por fim, um sistema híbrido consiste em combinar as duas abordagens 
    mencionadas, tentando fortificá-las e superar suas desvantagens.

    Dentro da categoria Filtragem Colaborativa, pode-se ainda dividir os
    sistemas em mais duas categorias: Item-Based e User-Based. Destaca-se
    nesse meio o algoritmo SlopeOne, do tipo Item-Based, uma abordagem
    simples e eficiente

    fonte: https://www.ibm.com/developerworks/br/local/data/sistemas_recomendacao/index.html

##### 3)
    Recent research has demonstrated that a hybrid approach, combining 
    collaborative filtering and content-based filtering could be more effective in
     some cases. Hybrid approaches can be implemented in several ways: by making 
     content-based and collaborative-based predictions separately and then 
     combining them; by adding content-based capabilities to a collaborative-based 
     approach (and vice versa); or by unifying the approaches into one model (see
     [21] for a complete review of recommender systems). Several studies 
     empirically compare the performance of the hybrid with the pure collaborative 
     and content-based methods and demonstrate that the hybrid methods can provide 
     more accurate recommendations than pure approaches. These methods can also be 
     used to overcome some of the common problems in recommender systems such as 
     cold start and the sparsity problem.

    Netflix is a good example of the use of hybrid recommender systems.[48] The 
    website makes recommendations by comparing the watching and searching habits of 
    similar users (i.e., collaborative filtering) as well as by offering movies 
    that share characteristics with films that a user has rated highly 
    (content-based filtering).

    A variety of techniques have been proposed as the basis for recommender 
    systems: collaborative, content-based, knowledge-based, and demographic 
    techniques. Each of these techniques has known shortcomings, such as the well 
    known cold-start problem for collaborative and content-based systems (what to 
    do with new users with few ratings) and the knowledge engineering bottleneck[49]
     in knowledge-based approaches. A hybrid recommender system is one that 
     combines multiple techniques together to achieve some synergy between them.


    Collaborative: The system generates recommendations using only information 
    about rating profiles for different users or items. Collaborative systems 
    locate peer users / items with a rating history similar to the current user
     or item and generate recommendations using this neighborhood. The user 
      deal with the cold start problem and improve recommendation results.[50]
    Content-based: The system generates recommendations from two sources: the 
    features associated with products and the ratings that a user has given 
    them. Content-based recommenders treat recommendation as a user-specific 
    classification problem and learn a classifier for the user's likes and 
    dislikes based on product features.
    Demographic: A demographic recommender provides recommendations based on a 
    different demographic niches, by combining the ratings of users in those 
    niches.
    Knowledge-based: A knowledge-based recommender suggests products based on 
    inferences about a user’s needs and preferences. This knowledge will 
    sometimes contain explicit functional knowledge about how certain product 
    features meet user needs.[51][52]

    The term hybrid recommender system is used here to describe any recommender 
    system that combines multiple recommendation techniques together to produce its
     output. There is no reason why several different techniques of the same type
      could not be hybridized, for example, two different content-based 
      recommenders could work together, and a number of projects have investigated 
      this type of hybrid: NewsDude, which uses both naive Bayes and kNN 
      classifiers in its news recommendations is just one example.[51]

    Seven hybridization techniques:

    Weighted: The score of different recommendation components are combined 
    numerically.
    Switching: The system chooses among recommendation components and applies 
    the selected one.
    Mixed: Recommendations from different recommenders are presented together 
    to give the recommendation.
    Feature Combination: Features derived from different knowledge sources are 
    combined together and given to a single recommendation algorithm.
    Feature Augmentation: One recommendation technique is used to compute a 
    feature or set of features, which is then part of the input to the next 
    technique.
    Cascade: Recommenders are given strict priority, with the lower priority 
    ones breaking ties in the scoring of the higher ones.
    Meta-level: One recommendation technique is applied and produces some sort 
    of model, which is then the input used by the next technique.[51]

    fonte: https://en.wikipedia.org/wiki/Recommender_system

## Características de cada paradigma
#### tipos
1. Sistemas colaborativos
2. Sistemas de Conteúdo
3. Sistemas sociais

(ver o topico a cima)

• Recomendação personalizada: recomenda coisas baseada no histórico
de comportamento
• Recomendação social: recomenda coisas baseada no histórico de
comportamento de usuários parecidos
• Recomendação de item: recomenda coisas baseada na própria coisa
• Uma combinação das três abordagens acima

## Vantagens para o promotor da recomendaçãoo
##### 1)
    Sistemas de “tagueamento”, como a folksonomia, apresentam duas
    desvantagens que são próprias da linguagem natural, ou seja,
    a linguagem que utilizamos no dia-a-dia. Sem um controle dos
    vocábulos, os usuários podem criar vários termos para o mesmo
    conceito: por exemplo, para o termo New York City, o usuário
    pode criar tags como NYC, Newyork ou Newyourkcity. A segunda
    desvantagem é quando palavras homógrafas representam conceitos
    diferentes. Para entender mais sobre esse tema, leia o artigo
    “Ecommerce 3.0: novos paradigmas e novos desafios impostos pela
    web do futuro – a Websemântica”, que foi publicado na revista de
    fevereiro e está disponível online em http://bit.ly/wFRywP.

    Por outro lado, a indexação com o uso de um vocabulário controlado
    (VC), ao contrário da folksonomia, não permite que o próprio usuário
    escreva ou crie as palavras-chave de forma “descontrolada”, pois elas
    devem ser selecionadas a partir de uma lista pré-definida e composta
    apenas por tags autorizadas. Nessa medida, o VC faz um contraponto
    com a folksonomia, porque esta usa a linguagem natural, e o VC
    usa um vocabulário que busca “controlar” as ambiguidades.

    Enquanto o VC nos “lembra” e nos “impõe” o uso de apenas um, de dois
    ou mais termos sinônimos e nos permite identificar a distinção conceitual
    entre palavras homógrafas, na folksonomia é o usuário quem cria
    livremente as suas tags preferidas. Assim, se a indexação ou “tagueamento”
    é feito a partir da linguagem natural do usuário, a comunicação
    está sujeita a “ruídos” que podem tornar ambíguo o conceito representado
    pelos “genes” ou tags. Aí, nesse momento, começam a surgir as imprecisões
    ou mal entendidos, sejam entre humanos ou entre homens e máquinas.

    Quanto mais coerentes as relações semânticas entre as tags ou conceitos,
    mais assertivas serão as recomendações ou buscas que se baseiam nesses
    “genes”. Uma abordagem que é híbrida, e que já é adotada pela Amazon,
    combina a liberdade de criação de tags com um processo de moderação
    que impõe ao cliente algumas regras, ou seja, são disponibilizadas
    funcionalidades que o permite organizar os produtos da loja à sua
    própria maneira, por meio das tags, bem como fazer buscas por produtos
    tagueados por outros clientes e até usar as tags para agrupar e comparar
    lado-a-lado produtos que considera comprar. Mas tudo isso de forma 
    “supervisionada”, para garantir a qualidade das tags e dos serviços que delas
    dependem.

    A Amazon não organizou um concurso de US$ 1 milhão, mas disponibiliza uma
    série de ferramentas sociais que viabilizam o trabalho que é pago através
    da troca de diferentes satisfações e necessidades pessoais. E são essas
    satisfações e necessidades sociais que despertam a motivação intrínseca
    de cada pessoa de uma horda de clientes para comentar a performance
    e as funcionalidades uma câmera fotográfica, a votar com estrelinhas
    em um livro que de gostou ou odiou, a sugerir novas tags e a votar
    nas tags existentes de um produto que comprou ou que deseja comprar.
    E, isso, US$ 1 milhão não paga… ou melhor, as lojas online é que
    ganham milhões.
    
    fonte: https://www.ecommercebrasil.com.br/artigos/tags-o-dna-dos-sistemas-de-recomendacao/

## Vantagens para o alvo da recomendação

## Técnicas de machine learning utilizadas
### Sistema de Recomendação slopeOne
##### 1)
    O Slope One é um método de Recomendação fácil de implementar, com teoria
    simples e que apresenta bons resultados práticos sendo altamente escalável
    [LEMIRE, Daniel. MACLACHLAN, Anna. Slope One Predictors for Online Rating-Based
    Collaborative Filtering, In SIAM Data Mining (SDM'05), Newport Beach,
    California, April 21-23, 2005 ].
    Apresentado em um artigo de Daniel Lemire em 2005, suas predições são
    calculadas a partir da comparação entre avaliações de usuários a certos
    itens.
    O algoritmo opera supondo que um usuário tenha dado notas não binárias
    a itens. Essas notas são colocadas em uma matriz de UsuáriosxItens,
    de tal forma que cada linha corresponda às notas de um usuário j a N itens.
    Se um usuário j não tiver dado notas a um item i, o elemento xi,j fica igual a 0.
    A figura representa a Matriz com um conjunto de notas.

    Observando a matriz de notas vemos que uma linha j da matriz representa
    as notas dadas por um usuário j a todos os itens no espaço definido.
    Uma coluna i Representa as notas recebidas pelo item i pelos diferentes
    usuários existentes.

    A partir dessa matriz, podemos obter relações entre os dados.
    É possível gerar uma interpolação matemática e predizer qual seria
    a nota dada por um usuário j ao item i que ele ainda não avaliou.

    A maioria dos métodos de Filtragem Colaborativa também utiliza a matriz
    de notas para calcular predições. Comumente, são calculas as similaridades
    entre as linhas ou colunas usando funções como Pearson ou Cosine Similarity.
    Dizemos que o método é User-Based quando são comparadas as linhas da matriz
    e Item-Based para colunas.

    Diferentemente de outras abordagens colaborativas, o Slope One cria uma relação
    linear entre os dados. Dai vem o nome: slope é o multiplicador de x na fórmula
    f(x) = ax + b, e o slope para esse algoritmo equivale a 1.

    Supondo que temos, um usuário A que deu nota 2 para um item i, e nota 4 a um 
    item j e supondo ainda que temos um usuário B que deu nota 3 para o item i,
    através do SlopeOne, calcularía-se a predição da nota que o usuário B daria ao
    item j da seguinte forma:

    De acordo com a predição do algoritmo, o usuário B daria uma nota 5 ao item j.

    Para análise de mais dados, obtería-se a média das diferenças entra as notas
    dos usuários. A fórmula geral para cálculo das predições segue descrita abaixo:

    Onde Diff (i,j) é a média das diferenças de avaliações entre itens i e j para
    os outros usuários, R (A, j) é quanto o usuário A deu de nota ao item j,
    e supondo que tenhamos N itens e que os itens variem de i a z .

    fonte: https://www.ibm.com/developerworks/br/local/data/sistemas_recomendacao/index.html

## Exemplos característicos
### Comercio eletronico 
1. Amazon
2. ebay
3. playstore
4. ...
--------------------
35% das vendas da Amazon vem de recomendações.
http://coral.ufsm.br/pet-si/index.php/sistemas-de-recomendacao-desvendando-uma-parte-da-magica/
--------------------

### Hotelaria e Restauracao
1. McDonald's
2. Booking 
3. Trivago