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
    Sistemas de Recomendação podem ser vistos ao realizar buscas em sites
    de pesquisa da internet, em compras online, ou até mesmo ao visualizamos
    nossos emails. São o mecanismo por trás da propaganda personalizada que
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

## Características de cada paradigma
#### tipos
1. Sistemas colaborativos
2. Sistemas de Conteúdo
3. Sistemas sociais

## Vantagens para o promotor da recomendação

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

### Hotelaria e Restauracao
1. McDonald's
2. ...