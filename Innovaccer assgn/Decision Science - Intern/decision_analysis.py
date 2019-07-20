
# coding: utf-8

# # Goli Dataset Analysis

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


market_ext = pd.read_csv('./Market Pricing.csv')
cust_bill_details = pd.read_csv('./Customer Bill Detail.csv')
cust_order_details = pd.read_csv('./Customer Order Item Details.csv')


# In[3]:


market_ext.head()


# In[4]:


cust_bill_details.head()


# In[5]:


cust_order_details.head()


# In[6]:


market_ext.sample(20)


# In[7]:


market_ext.count()


# In[8]:


cust_bill_details.count()


# In[9]:


cust_order_details.count()


# In[10]:


cust_order_details.groupby('Bill Number').count().shape


# Let's unify the date time formats.

# In[11]:


cust_order_details.dtypes


# In[12]:


cust_bill_details.dtypes


# In[13]:


cust_order_details.sample()


# In[14]:


cust_bill_details.sample()


# In[15]:


cust_order_details.Date = pd.to_datetime(cust_order_details.Date, format="%d-%m-%Y")


# In[16]:


cust_bill_details.Date = pd.to_datetime(cust_bill_details.Date, format="%Y-%m-%d")


# In[17]:


cust_order_details.dtypes


# In[18]:


cust_bill_details.dtypes


# In[19]:


cust_order_details.sample()


# In[20]:


cust_bill_details.sample()


# In[21]:


cust_order_details.Date.max()


# In[22]:


cust_bill_details.Date.max()


# Data is of the same timeline.

# Let's first merge the bill and order details into a single table.

# In[23]:


total_details = pd.merge(cust_order_details, cust_bill_details, left_on=['Date','Bill Number'],
                         right_on=['Date','Bill Number'])


# In[24]:


total_details.head()


# In[25]:


total_details.dtypes


# In[26]:


g = total_details.groupby(["Date",'Bill Number'])
for key, value in g:
    print(g.get_group(key))


# In[27]:


total_details['Dish Details'][0]


# ## Competitive Pricing

# For competitive pricing, let's analyse the Market Pricing Dataset first, and try to see the average prices of the competition.

# In[28]:


market_ext.count()


# Some *Rate* columns, and a majority of loved dishes data is missing. Let's see the restaurants they correspond to.

# In[29]:


market_ext[market_ext.Rate.isna()]


# These entries can simply be removed.

# In[30]:


market_ext = market_ext.dropna(subset=['Rate'])


# In[31]:


market_ext.Restaurant.unique() #35


# In[32]:


list(market_ext[market_ext.Restaurant == 'Goli']['Menu Item'])


# In[33]:


market_ext[market_ext.Restaurant == 'Goli']['Menu Header'].unique()


# In[34]:


market = market_ext.copy()


# To ensure uniformity, let's correspond all menu headers to that of Goli's.

# In[35]:


#Starters and Tandoori Starters
market['Menu Header'] = market_ext['Menu Header'].replace(to_replace='(?i)^.*starters.*$',value='Starters', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^starters.*$',value='Starters', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*starters$',value='Starters', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*sizzler.*$',value='Starters', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*soup.*$',value='Starters', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*appetizer.*$',value='Starters', regex=True)


# In[36]:


#Combos
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*combos.*$',value='Combos', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*conbos.*$',value='Combos', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*combo.*$',value='Combos', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*thali.*$',value='Combos', regex=True)


# In[37]:


#Noodles
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*noodle.*$',value='Fried Rice and Noodles', regex=True)


# In[38]:


#Rice and Biryani
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*rice.*$',value='Rice and Biryani', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^rice.*$',value='Rice and Biryani', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*rice$',value='Rice and Biryani', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*biryani.*$',value='Rice and Biryani', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*pulao.*$',value='Rice and Biryani', regex=True)


# In[39]:


#Chinese Dishes
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*chinese.*$',value='Chinese Dishes', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*chineese.*$',value='Chinese Dishes', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*chineese$',value='Chinese Dishes', regex=True)


# In[40]:


#Breads
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*bread.*$',value='Breads', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*roti.*$',value='Breads', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*naan.*$',value='Breads', regex=True)


# In[41]:


#Accompaniments
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*papad.*$',value='Accompaniments', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*salad.*$',value='Accompaniments', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*raita.*$',value='Accompaniments', regex=True)


# In[42]:


#Kebabs
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*kabab.*$',value='Kebabs', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*kabeb.*$',value='Kebabs', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*kebab.*$',value='Kebabs', regex=True)


# In[43]:


#Main Course
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*main.course.*$',value='Main Course', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*dal.*$',value='Main Course', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*daal.*$',value='Main Course', regex=True)


# In[44]:


#Platters
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*platter.*$',value='Platters', regex=True)


# In[45]:


#Rolls
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*roll.*$',value='Rolls', regex=True)


# In[46]:


#Breakfast
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*breakfast.*$',value='Breakfast', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*paratha.*$',value='Breakfast', regex=True)
market['Menu Header'] = market['Menu Header'].replace(to_replace='(?i)^.*curd.*$',value='Breakfast', regex=True)


# We ignore beverages, desserts/ice-creams, liquors, snacks and fast food, since they are not sold by Goli's.

# In[47]:


market.loc[market['Menu Item'].str.contains('(?i).*soup.*', regex=True),'Menu Header'] = 'Starters'


# In[48]:


market.loc[market['Menu Item'].str.contains('(?i).*combo.*', regex=True),'Menu Header'] = 'Combos'
market.loc[market['Menu Item'].str.contains('(?i).*thali.*', regex=True),'Menu Header'] = 'Combos'


# In[49]:


market.loc[market['Menu Item'].str.contains('(?i).*noodle.*', regex=True),'Menu Header'] = 'Noodles'


# In[50]:


market.loc[market['Menu Item'].str.contains('(?i).*rice.*', regex=True),'Menu Header'] = 'Rice and Biryani'
market.loc[market['Menu Item'].str.contains('(?i).*biryani.*', regex=True),'Menu Header'] = 'Rice and Biryani'
market.loc[market['Menu Item'].str.contains('(?i).*pulao.*', regex=True),'Menu Header'] = 'Rice and Biryani'


# In[51]:


market.loc[market['Menu Item'].str.contains('(?i).*chinese.*', regex=True),'Menu Header'] = 'Chinese Dishes'
market.loc[market['Menu Item'].str.contains('(?i).*manchurian.*', regex=True),'Menu Header'] = 'Chinese Dishes'


# In[52]:


market.loc[market['Menu Item'].str.contains('(?i).*roti.*', regex=True),'Menu Header'] = 'Breads'
market.loc[market['Menu Item'].str.contains('(?i).*naan.*', regex=True),'Menu Header'] = 'Breads'


# In[53]:


market.loc[market['Menu Item'].str.contains('(?i).*papad.*', regex=True),'Menu Header'] = 'Accompaniments'
market.loc[market['Menu Item'].str.contains('(?i).*salad.*', regex=True),'Menu Header'] = 'Accompaniments'
market.loc[market['Menu Item'].str.contains('(?i).*raita.*', regex=True),'Menu Header'] = 'Accompaniments'


# In[54]:


market.loc[market['Menu Item'].str.contains('(?i).*kebab.*', regex=True),'Menu Header'] = 'Kebabs'
market.loc[market['Menu Item'].str.contains('(?i).*kabab.*', regex=True),'Menu Header'] = 'Kebabs'
market.loc[market['Menu Item'].str.contains('(?i).*kabeb.*', regex=True),'Menu Header'] = 'Kebabs'


# In[55]:


market.loc[market['Menu Item'].str.contains('(?i).*dal.*', regex=True),'Menu Header'] = 'Main Course'


# In[56]:


market.loc[market['Menu Item'].str.contains('(?i).*roll.*', regex=True),'Menu Header'] = 'Rolls'


# In[57]:


market.loc[market['Menu Item'].str.contains('(?i).*paratha.*', regex=True),'Menu Header'] = 'Breakfast'
market.loc[market['Menu Item'].str.contains('(?i).*curd.*', regex=True),'Menu Header'] = 'Breakfast'


# We could go on with the categorisation by further classifying snacks and liquors, and then getting more meals for the Main Course, but we would halt it here for now, and compare prices of Goli and its competition.
# //TODO: Add Veg/NonVeg flags.

# In[58]:


CATEGORIES = ['Main Course', 'Kebabs', 'Rice and Biryani', 'Platters',
       'Breakfast', 'Starters', 'Deep Fried', 'Breads', 'Rolls',
       'Chinese Dishes', 'Accompaniments']


# In[59]:


market.dtypes


# In[60]:


market.Rate.unique()


# We see that there is a stray 'MRP' value, which we can remove, as well as price variations(such as Bournvita vs plain milk) divided by a slash. We will consider the average price for the dish in that case.

# In[61]:


market.loc[market.Rate == 'MRP', 'Rate'] = '30' #Average
market = market[market.Rate != 'NOT FOUND']
market.loc[market.Rate == '25/30/40/40/50', 'Rate'] = '40'


# In[62]:


slashed_prices = market.Rate.str.extract('(.*)/(.*)')


# In[63]:


slashed_prices.count()


# In[64]:


slashed_prices[~slashed_prices[0].isna()]


# In[65]:



slash_prices = (slashed_prices[~slashed_prices[0].isna()][0].astype(int) + slashed_prices[~slashed_prices[0].isna()][1].astype(int)) / 2


# In[66]:


slash_prices


# In[67]:


market.loc[~slashed_prices[0].isna(),'Rate'] = slash_prices


# In[68]:


market.Rate =market.Rate.astype(np.float64)


# Now we can plot the mean data and see where Goli stands in comparison to its competitors.

# In[69]:


g = market.groupby(['Menu Header', 'Restaurant']).mean()
g


# In[70]:


market[(market.Restaurant == 'Foot') & (market['Menu Header'] == 'Main Course')]


# In[71]:


def plot_func(g):
    if(g['Menu Header'].unique() in CATEGORIES):
        h = g.groupby('Restaurant').mean()
        plt.figure()
        plt.tick_params(rotation=90)
        sns.scatterplot(x=h.index, y=h.Rate)
        plt.title(g['Menu Header'].unique())
g = market.groupby('Menu Header').apply(plot_func)


# In[72]:


market[(market.Restaurant == 'Foot') & (market['Menu Header'] == 'Main Course')]


# In[73]:


market[(market.Restaurant == 'Pisa') & (market['Menu Header'] == 'Main Course')]


# We see that Goli is already priced quite competitively to its competition in all categories. The price in Main Course looks higher, but that's only because restaurants like Pisa and Foot don't place their non vegetarian dishes in the Main Course section, while Goli does.

# Conclusion: Goli is already priced quite competitively in all the segments it sells. However, to increase sales, there is a lot of untapped potential in categories such as Beverages, Desserts, and Snacks. Liquors could be explored if they wish to garner a different kind of audience. 

# Beverages is quite an ironical case, since people seem to order them anyway, but they aren't listed in the menu. This can be explored further in upcoming sections.

# ## Dishes to promote and remove

# In[74]:


cust_menu = pd.merge(cust_order_details, market[market.Restaurant == 'Goli'], left_on='Dish Name', right_on='Menu Item',
                     how='left')


# In[75]:


cust_menu


# Let's look at the dishes by the quantity at which they're sold.

# In[76]:


sellers = cust_menu.groupby('Dish Name').sum()['Quantity']


# Let's look at the top 20 bestsellers.

# In[77]:


top20 = sellers.sort_values(ascending=False)[:20]


# In[78]:


plt.figure(figsize=(10,10))
plt.tick_params(rotation=0)
sns.barplot(y=top20.index, x=top20,color='brown')


# If we remove the breads and just put main course dishes on the graph, it's clear that Goli is basically a very non vegetarian centric restaurant, and that they should promote their marquee dishes such as  Hyderabadi Chicken Fry Piece Biryani, Chicken Curry, Butter Chicken the most. However, since this data is already present in the database, we're pretty sure Goli already knows this.

# What is a bigger surprise is that beverages such as Cold Drinks, water bottles, etc sell like hot cakes in Goli's, yet they are not even mentioned in the menu. We think Goli could make a decent chunk of profit if he mentions it in the menu, and then sells them a notch above their price.

# Let's now look at the worst sellers.

# Since we have the data of 4 months(July to September), it is reasonable to expect that any dish that sells less than once a month is a waste to keep. So:

# In[79]:


worst_sellers = sellers[sellers <= 5]


# In[80]:


worst_sellers2 = sellers[sellers <= 10]


# In[81]:


worst_sellers.sort_values()


# The above list contains strong candidates for being removed. However, this data has a lot of anomalies.

# In[82]:


goli_info = market[market.Restaurant == 'Goli']


# In[83]:


goli_info.loc[goli_info['Menu Item'].str.contains('(?i).*burger.*', regex=True),'Menu Item'] 


# In[84]:


goli_info.loc[goli_info['Menu Item'].str.contains('(?i).*tea.*', regex=True),'Menu Item'] 


# In[85]:


goli_info.loc[goli_info['Menu Item'].str.contains('(?i).*noodles.*', regex=True),'Menu Item'] 


# In[86]:


goli_info.loc[goli_info['Menu Item'].str.contains('(?i).*lassi.*', regex=True),'Menu Item'] 


# In[87]:


goli_info.loc[goli_info['Menu Item'].str.contains('(?i).*sev.*', regex=True),'Menu Item'] 


# ... And so on.

# What we observe, is that there are many items in the above worst sellers list, such as Green tea, Singapuri Noodles, Chocolate shake, Lassi, as well as many combos(such as the Sev Tamatar combo) that aren't even LISTED in the menu. How does Goli expect its customers to know they make them?

# This is a big anomaly in the above data, and hence we can't definitively say which dishes should be removed from the menu since most of them aren't being advertised. Let's take a look at the dishes that don't perform well even after being advertised in the menu:

# In[88]:


pd.merge(pd.DataFrame(worst_sellers2), goli_info, left_index=True, right_on='Menu Item', how='left').dropna()


# These are the dishes don't sell well even after being in the menu(less than 10 orders in 4 months), and hence seem like better candidates to be removed from the menu.

# ## Customer Segments

# #### By how much they spend

# In[89]:


cust_bill_details


# In[90]:


cust_bill_details.groupby(['Type','Channel']).mean()


# In[91]:


cust_bill_details.groupby(['Type', 'Channel']).count()


# In[92]:


sns.countplot(data=cust_bill_details, x='Type')
plt.title('Types of customers based on ordering place')


# In[93]:


sns.countplot(data=cust_bill_details, x='Channel')
plt.title('Types of customers based on Channel')


# In[94]:


typedat = cust_bill_details.groupby('Type').mean()


# In[95]:


sns.barplot(x=typedat.index, y=typedat.Discount)
plt.title('Discounts')


# In[96]:


sns.barplot(x=typedat.index, y=typedat['Total Bill'])
plt.title('Total bill by each type')


# * As we clearly see from the graphs, as well as the tables above, Home delivery is the most popular way of ordering at Goli. This is generally because of the great discounts at offer from both Zomato and Swiggy, and Swiggy is the clear winner at discounted food. However, Zomato is the most popular way of ordering, despite reduced discounts.

# * There is a clear correlation why Home Delivery is popular. Because it's cheap!
# We even see some Dine In and Takeaway customers ordering through Zomato and Swiggy because it's cheap. This points to the extreme competitiveness of online pricing.

# * One advice to Goli: We see that their charges on Delivery are the highest as compared to Swiggy and Zomato. This would deter people away from calling and ordering directly through Goli and would surely hurt their profits. Hence, they should reduce their delivery charges.

# Let's plot a histogram of customer spending.

# In[97]:


sns.distplot(cust_bill_details['Total Bill'])


# We see that we can easily divide the consumers between 0-300 spenders, 300-1000 spenders, and 1000+ spenders.

# **Let's check the 1000+ spenders first.**

# In[98]:


(cust_bill_details['Total Bill'] > 1000).value_counts()


# Only 110.

# In[99]:


sns.countplot(x=cust_bill_details[cust_bill_details['Total Bill'] > 1000].Type)


# Most of the heavy spenders are basically Take away type orders.

# **300-1000**

# In[100]:


((cust_bill_details['Total Bill'] < 1000) & (cust_bill_details['Total Bill'] > 300)).value_counts()


# In[101]:


sns.countplot(x=cust_bill_details[(cust_bill_details['Total Bill'] < 1000) & (cust_bill_details['Total Bill'] > 300)].Type)


# As expected, most of the midway spenders are the home delivery people.

# **0-300**

# In[102]:


(cust_bill_details['Total Bill'] <= 300).value_counts()


# In[103]:


sns.countplot(x=cust_bill_details[(cust_bill_details['Total Bill'] <= 300)].Type)


# The less than 300 audience is the majority spender(around 6000 out of a total of 8000), and as expected, they order heavily from online sources and are home delivery. So this is the audience that Goli has to cater the most.

# #### By what they order

# In[104]:


cust_order_details


# We can divide the customers in Veg and Non-Veg, whether they order appetizers or not, whether they order drinks or not, and by the quantity they order.

# In[105]:


cust_order = cust_order_details.copy()


# In[106]:


cust_order['is Non-Veg'] = np.where(cust_order['Dish Name'].str.contains('(?i).*chicken.*|.*mutton.*|.*non veg.*|.*egg.*', regex=True), 1, 0)


# In[107]:


cust_order['is appetizer'] = np.where(cust_order['Dish Name'].str.contains('(?i).*salad.*|.*papad.*|.*raita.*', regex=True), 1, 0)


# In[108]:


cust_order['is a drink'] = np.where(cust_order['Dish Name'].str.contains('(?i).*drink.*|.*coffee.*|.*tea.*|.*lassi.*|.*water.*', regex=True), 1, 0)


# In[109]:


cust_data = cust_order.groupby('Bill Number').sum()


# In[110]:


sns.countplot(data=cust_data, x='is Non-Veg')
plt.title('Number of Non-Veg items in order')


# In[111]:


sns.countplot(data=cust_data, x='is appetizer')
plt.title('Number of Appetizers in order')


# In[112]:


sns.countplot(data=cust_data, x='is a drink')
plt.title('Number of Drinks in order')


# In[113]:


plt.figure(figsize=(30,10))
sns.boxplot(data=cust_data, x='Quantity')
plt.title('Number of items in order')


# * Clearly, we have a majority of people who order at least one non vegetarian item in their orders, as opposed to the vegetarian kind(we include egg eaters as non-vegetarians).

# * There are a considerable number of drinks in the orders, even though they are not on the menu. This again points to their need in the menu.

# * Maximum number of orders have a maximum order length of 10, barring some special customers.

# Hence, we have characterised customers based on both what they order, how they order, as well as their spendings.

# ## Most loved dishes

# In[114]:


loved_dishes = market.groupby('Restaurant').head(1).dropna()['What people love here']


# In[115]:


list_of_loved_dishes = loved_dishes.str.split(',', expand=True).stack()


# In[116]:


sns.countplot(x=list_of_loved_dishes)


# This has too many 1s and 2s, let's filter them out.

# In[117]:


top9 = list_of_loved_dishes.value_counts().sort_values(ascending=False)[:9]


# In[118]:


plt.figure(figsize=(10,10))
plt.tick_params(rotation=45)
sns.barplot(x=top9.index, y=top9)
plt.title('Most loved dishes')


# Clearly, Butter Chicken is the most loved of them all.

# ## Combo Analysis

# We can't do a combo analysis by examining rival combo data, since we don't know how much those rival combos sell. So, we only examine our own data.

# In[119]:


cust_order


# As we have analyzed before:

# In[120]:


sns.countplot(data=cust_data, x='is Non-Veg')
plt.title('Number of Non-Veg items in order')


# In[121]:


sns.countplot(data=cust_data, x='is a drink')
plt.title('Number of Drinks in order')


# By this, we feel drinks such as cold drinks(proven to be good sellers) can be bundled with Main Course meals as combos at a price around 10-20 rupees less than their original MRP.

# ## Strategies to Increase Revenue

# We have already discussed quite a few strategies to increase revenue. Let's sum them up again:

# * They should expand their portfolio of dishes. We see the competition selling in categories like Snacks, Beverages, Desserts, while Goli isn't even featuring them in their menu. 
# * We have already identified the best and worst selling dishes, and the worst selling dishes should be removed, since they drag the reputation of the restaurant down. The best dishes have already been identified in the dataset('What people love here'), which is a good thing. However, there are many dishes that are being sold from Goli's(such as soft drinks, lassi, noodles, combos) that are not even mentioned in their menu. If they are mentioned, they would surely boost sales, and can be sold a bit above their original price.
# * The heavy spenders at Goli's generally prefer Take away orders, so Goli should give them minimum spending incentives(like buy food of ₹1000, and get a meal coupon for the next meal, etc) so that they keep spending big at Goli's.
# * Goli has most of its customers using Home delivery, so it should put offers at those platforms to boost sales.
# * Affordable combos for the ₹0-300 spenders would result in faster conversions, since these people generally look for combos that serve 1 or 2 people.
# * Every restaurant should focus on improving their quality of Non-Veg products, since they contribute the most to revenue.

# For testing the last point:

# In[122]:


cust_data_details = pd.merge(cust_bill_details, cust_data, left_on='Bill Number', right_index=True, how='right')


# In[123]:


cust_data_details


# In[124]:


veg_orders = cust_data_details[cust_data_details['is Non-Veg'] == 0]
non_veg_orders = cust_data_details[cust_data_details['is Non-Veg'] > 0]


# In[125]:


veg_orders['Total Bill'].describe()


# In[126]:


plt.figure(figsize=(20,5))
ax1 = plt.subplot(2,1,2)
sns.boxplot(non_veg_orders['Total Bill'])

ax2 = plt.subplot(2,1,1, sharex=ax1)
sns.boxplot(veg_orders['Total Bill'])


# In[127]:


non_veg_orders['Total Bill'].describe()


# As we see clearly, not only are Non-Veg orders 3/4ths of the total orders at Goli's, they also are costlier on average by a hefty sum of ₹100, and have greater min-max values.

# We could also have a check on recurring customers to see their trends:

# In[135]:


cust_data_details['Customer ID'].value_counts().sort_values(ascending=False)


# There is a certain sense of fallacy here, since some customer IDs have recurrence values in hundreds. Is it possible that a customer orders ~600 times in a span of 4 months?

# In[136]:


cust_data_details[cust_data_details['Customer ID'] == '38271357206']


# This is the most frequent Customer ID, and it has 689 entries within a span of 4 months. This data looks dubious, at best.

# Let's look at some other customer ID.

# In[137]:


cust_data_details[cust_data_details['Customer ID'] == '289200068721']


# Same story. We see that the same customer ID orders 96 times in a month. This is highly unreliable data. Or it might be that someone is using the same ID to order for different people. In any case, we cannot make any trends or conclusions out of such data.

# Let's try a relatively smaller number and see what that brings.

# In[138]:


cust_data_details[cust_data_details['Customer ID'] == '258819372616']


# This seems like a regular profile that seems to order semi regularly from Goli's. However, there is no visible trend in the quantity order or the bill amount, that we can observe.

# From this customer ID analysis, we have two points to keep in mind:
# * Goli should come up with a way to catch same customer ID, or encourage customers not to reuse IDs by first time offers, etc
# * The genuine highest frequent customers should be rewarded by discount coupons, etc

# # Conclusion

# We have hence analyzed the dataset for various factors, and in the end have suggested some valuable insights for boosting Goli's business.
