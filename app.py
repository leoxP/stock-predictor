from flask import Flask,render_template,request

app=Flask(__name__,template_folder="templates",static_folder="static")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/search', methods=['POST'])
def search():
    stock_symbol=request.form.get('searchBar')
    print(f'Stock Symbol: {stock_symbol}')
    return render_template("stock-details.html")

@app.route('/back',methods=['POST'])
def back():
    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)
