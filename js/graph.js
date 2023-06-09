let rowConverter = function(d) {
    let date = []
    let open = []
    let high = []
    let low = []
    let adj_Close = []
    let volume = []
    let close = []

    // Date,Open,High,Low,Close,Adj_Close,Volume  d.length
    for (let i = 0; i < d.length; i++) {
        open.push(Number.parseFloat(d[i].Open))
        high.push(Number.parseFloat(d[i].High))
        low.push(Number.parseFloat(d[i].Low))
        adj_Close.push(Number.parseFloat(d[i].Adj_Close))
        volume.push(Number.parseFloat(d[i].Volume))
        date.push(d[i].Date)
        close.push((Number.parseFloat(d[i].Close)))
    }
    return {
        date,
        open,
        high,
        low,
        adj_Close,
        volume,
        close
    }
}

let drawCandleSimple = (idDiv, data, title) => {
    let trace = {
        x: data.date,
        date: data.date,
        close: data.close,
        high: data.high,
        low: data.low,
        open: data.open,
        volume: data.volume,
        name: 'Candlestick',
        increasing: {
            line: {
                color: 'black'
            }
        },
        decreasing: {
            line: {
                color: '#red'
            }
        },

        type: 'candlestick',
        //type: typeChart,
        xaxis: 'x',
        yaxis: 'y'
    };

    let input = [trace];
    let layout = {
        dragmode: 'zoom',
        showlegend: true,

        xaxis: {
            autorange: true,
            title: title,

            rangeslider: {
                visible: false
            },
            rangeselector: {
                x: 0,
                y: 1.2,
                xanchor: 'left',
                font: {
                    size: 8
                },
                buttons: [{
                    count: 1,
                    label: '1m',
                    step: 'month',
                    stepmode: 'backward'
                }, {
                    count: 6,
                    label: '6m',
                    step: 'month',
                    stepmode: 'backward'
                }, {
                    step: 'all'
                }]
            }
        },

        yaxis: {
            title: 'USD Price',

        },
        yaxis2: {
            title: 'Volume',
            overlaying: 'y',
            side: 'right',

        },
        yaxis_autorange: true,
        annotations: [
            /* {
                                    x: '2021-02-25',
                                    y: data.close[data.date.indexOf('2021-02-25')],
                                    xref: 'x',
                                    yref: 'y',
                                    text: ' US government bonds increased to 1.614%',
                                    showarrow: true,
                                    font: {
                                        family: 'Courier New, monospace',
                                        size: 16,
                                        color: '#ffffff'
                                    },
                                    align: 'center',
                                    arrowhead: 2,
                                    arrowsize: 1,
                                    arrowwidth: 2,
                                    arrowcolor: '#636363',
                                    ax: -220,
                                    ay: -150,
                                    bordercolor: '#c7c7c7',
                                    borderwidth: 2,
                                    borderpad: 4,
                                    bgcolor: '#ff7f0e',
                                    opacity: 0.8
                                }, */
            {
                x: '2021-12-01',
                y: data.close[data.date.indexOf('2021-12-01')],
                xref: 'x',
                yref: 'y',
                text: 'Signal Ukraine-Russia war',
                showarrow: true,
                font: {
                    family: 'Courier New, monospace',
                    size: 16,
                    color: '#ffffff'
                },
                align: 'center',
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 2,
                arrowcolor: '#636363',
                ax: -120,
                ay: -150,
                bordercolor: '#c7c7c7',
                borderwidth: 2,
                borderpad: 4,
                bgcolor: '#ff7f0e',
                opacity: 0.8
            }, {
                x: '2015-01-01',
                y: data.close[data.date.indexOf('2015-01-01')],
                xref: 'x',
                yref: 'y',
                text: 'Mt.Gox hack in early 2014',
                showarrow: true,
                font: {
                    family: 'Courier New, monospace',
                    size: 16,
                    color: '#ffffff'
                },
                align: 'center',
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 2,
                arrowcolor: '#636363',
                ax: 120,
                ay: -100,
                bordercolor: '#c7c7c7',
                borderwidth: 2,
                borderpad: 4,
                bgcolor: '#ff7f0e',
                opacity: 0.8
            }, {
                x: '2020-12-01',
                y: data.close[data.date.indexOf('2020-12-01')],
                xref: 'x',
                yref: 'y',
                text: 'The COVID-19',
                showarrow: true,
                font: {
                    family: 'Courier New, monospace',
                    size: 16,
                    color: '#ffffff'
                },
                align: 'center',
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 2,
                arrowcolor: '#636363',
                ax: -200,
                ay: -150,
                bordercolor: '#c7c7c7',
                borderwidth: 2,
                borderpad: 4,
                bgcolor: '#ff7f0e',
                opacity: 0.8
            },
        ]

    };
    Plotly.newPlot(idDiv, input, layout);





    var volumeTrace = {
        x: data.date,
        y: data.volume,
        type: 'line',
        name: 'Volume',
        yaxis: 'y2',
        marker: {

            color: '#5276EE' // Specify the desired color here
        }
    };


    Plotly.addTraces(idDiv, volumeTrace);


}

let drawBarChart = (row, column1, column2, name1, name2, idDiv) => {
    let trace1 = {
        x: row,
        y: column1,
        name: name1,
        type: "bar"
    }
    let trace2 = {
        x: row,
        y: column2,
        name: name2,
        type: "bar",

    }
    var data = [trace1, trace2];
    var layout = {
        barmode: "group",
        xaxis: {
            autorange: true,
            range: ['2015-02-17', '2017-02-16'],
            rangeselector: {
                buttons: [{
                    count: 1,
                    label: '1m',
                    step: 'month',
                    stepmode: 'backward'
                }, {
                    count: 6,
                    label: '6m',
                    step: 'month',
                    stepmode: 'backward'
                }, {
                    step: 'all'
                }]
            },
            rangeslider: {
                range: ['2015-02-17', '2017-02-16']
            },
            type: 'date'
        },
        yaxis: {
            autorange: true,
            range: [0, 138.870004167],
            type: 'linear'
        }
    };
    Plotly.newPlot(idDiv, data, layout);


}
let drawLine = (x, y, idDiv, title) => {
    let data = {
        type: "scatter",
        mode: "lines",
        name: title,
        x,
        increasing: {
            line: {
                color: 'black'
            }
        },
        decreasing: {
            line: {
                color: 'red'
            }
        },
        y
    }
    input = [data]
    let layout = {
        dragmode: 'zoom',
        showlegend: true,
        xaxis: {
            autorange: true,
            title: title,
            rangeselector: {
                x: 0,
                y: 1.2,
                xanchor: 'left',
                font: {
                    size: 8
                },
                buttons: [{
                    step: 'month',
                    stepmode: 'backward',
                    count: 1,
                    label: '1 month'
                }, {
                    step: 'month',
                    stepmode: 'backward',
                    count: 6,
                    label: '6 months'
                }, {
                    step: 'all',
                    label: 'All dates'
                }]
            }
        },
        yaxis: {
            autorange: true,
        }
    };

    Plotly.newPlot(idDiv, input, layout);

}
let addClickAnimation = (idDiv) => {
    let plot = document.getElementById(idDiv)
    plot.on('plotly_click', (data) => {
        let pts = '';
        for (let i = 0; i < data.points.length; i++) {
            annotate_text = 'x = ' + data.points[i].x +
                'y = ' + data.points[i].y.toPrecision(4);

            annotation = {
                text: annotate_text,
                x: data.points[i].x,
                y: parseFloat(data.points[i].y.toPrecision(4))
            }

            annotations = self.layout.annotations || [];
            annotations.push(annotation);
            Plotly.relayout('myDiv', {
                annotations: annotations
            })
        }
    })
}
let drawtwoline1 = (idDiv, x, column1, column2, name1, name2, title) => {
    let max1 = column1.reduce(function(accumulator, element) {
        return (accumulator > element) ? accumulator : element
    });
    let min1 = column1.reduce(function(accumulator, element) {
        return (accumulator < element) ? accumulator : element
    });

    let max2 = column2.reduce(function(accumulator, element) {
        return (accumulator > element) ? accumulator : element
    });
    let min2 = column2.reduce(function(accumulator, element) {
        return (accumulator < element) ? accumulator : element
    });


    let button_layer_2_height = 1
    let trace1 = {
        x,
        y: column1,
        mode: 'lines',
        type: 'scatter',
        name: name1,
        line: {
            color: 'red'
        }
    }

    let trace2 = {
        x,
        y: column2,
        mode: 'lines',
        type: 'scatter',
        name: name2,
        line: {
            color: '#7F7F7F'
        }
    }
    let column1_annotations = [{
        text: 'high ' + max1,
        x: x[column1.indexOf(max1)],
        y: max1,
        yref: 'y',
        xref: 'x',
        ay: 90,
        ax: 0
    }, {
        text: 'low ' + min1,
        x: x[column1.indexOf(min1)],
        y: min1,
        yref: 'y',
        xref: 'x',
        ay: 50,
        ax: 0
    }]
    let column2_annotations = [{
        text: 'High' + max2,
        x: x[column2.indexOf(max2)],
        y: max2,
        yref: 'y',
        xref: 'x',
        ay: -50,
        ax: 0
    }, {
        text: 'Low ' + min2,
        x: x[column2.indexOf(min2)],
        y: min1,
        yref: 'y',
        xref: 'x',
        ay: -50,
        ax: 0
    }]
    var updatemenus = [{
            buttons: [{
                    args: [{
                        'visible': [true, false, false]
                    }, {
                        'title': name1,
                        'annotations': column1_annotations
                    }],
                    label: name1,
                    method: 'update'
                }, {
                    args: [{
                        'visible': [false, true, false, ]
                    }, {
                        'title': name2,
                        'annotations': column2_annotations
                    }],
                    label: name2,
                    method: 'update'
                }, {
                    args: [{
                        'visible': [true, true, false, ]
                    }, {
                        'title': name1 + name2,
                        'annotations': [...column1_annotations, ...column2_annotations]
                    }],
                    label: 'Both',
                    method: 'update'
                }

            ],
            direction: 'left',
            pad: {
                'r': 10,
                't': 10
            },
            showactive: true,
            type: 'buttons',
            x: 0.1,
            xanchor: 'left',
            y: button_layer_2_height,
            yanchor: 'top'
        },

    ]
    let data = [trace1, trace2]
    let layout = {
        title,
        updatemenus: updatemenus,
        showlegend: false,
        xaxis: {
            autorange: true,
            range: ['2015-02-17', '2017-02-16'],
            rangeselector: {
                buttons: [{
                    count: 1,
                    label: '1m',
                    step: 'month',
                    stepmode: 'backward'
                }, {
                    count: 6,
                    label: '6m',
                    step: 'month',
                    stepmode: 'backward'
                }, {
                    step: 'all'
                }]
            },
            rangeslider: {
                range: ['2015-02-17', '2017-02-16']
            },
            type: 'date'
        },
        yaxis: {
            autorange: true,
            range: [86.8700008333, 138.870004167],
            type: 'linear'
        }
    };
    Plotly.newPlot(idDiv, data, layout);

}
let drawtwoline = (idDiv, data, title) => {
    // Create the trace for the area chart
    let trace6 = {
        x: data.date,
        y: data.close,
        // Fill area below the line
        name: 'Closing price',
        type: "scatter",
        mode: "lines",
        line: {
            color: "red"
        }, // Customize line color
    };

    // Create the layout for the area chart
    let layout = {
        dragmode: 'zoom',
        showlegend: true,
        title: title,
        xaxis: {
            autorange: true,
            title: "Date",
            rangeslider: {
                visible: false
            },
            rangeselector: {
                x: 0,
                y: 1.2,
                xanchor: 'left',
                font: {
                    size: 8
                },
                buttons: [{
                    count: 1,
                    label: '1m',
                    step: 'month',
                    stepmode: 'backward'
                }, {
                    count: 6,
                    label: '6m',
                    step: 'month',
                    stepmode: 'backward'
                }, {
                    step: 'all'
                }]
            }

        },
        yaxis: {
            title: "USD Price"
        },
        annotations: [{
            x: '2021-12-01',
            y: data.close[data.date.indexOf('2021-12-01')],
            xref: 'x',
            yref: 'y',
            text: 'Signal Ukraine-Russia war',
            showarrow: true,
            font: {
                family: 'Courier New, monospace',
                size: 16,
                color: '#ffffff'
            },
            align: 'center',
            arrowhead: 2,
            arrowsize: 1,
            arrowwidth: 2,
            arrowcolor: '#636363',
            ax: -120,
            ay: -150,
            bordercolor: '#c7c7c7',
            borderwidth: 2,
            borderpad: 4,
            bgcolor: '#ff7f0e',
            opacity: 0.8
        }, {
            x: '2015-01-01',
            y: data.close[data.date.indexOf('2015-01-01')],
            xref: 'x',
            yref: 'y',
            text: '2014-2015:Mt.Gox hack in early 2014',
            showarrow: true,
            font: {
                family: 'Courier New, monospace',
                size: 16,
                color: '#ffffff'
            },
            align: 'center',
            arrowhead: 2,
            arrowsize: 1,
            arrowwidth: 2,
            arrowcolor: '#636363',
            ax: 120,
            ay: -100,
            bordercolor: '#c7c7c7',
            borderwidth: 2,
            borderpad: 4,
            bgcolor: '#ff7f0e',
            opacity: 0.8
        }, {
            x: '2020-12-01',
            y: data.close[data.date.indexOf('2020-12-01')],
            xref: 'x',
            yref: 'y',
            text: 'The COVID-19',
            showarrow: true,
            font: {
                family: 'Courier New, monospace',
                size: 16,
                color: '#ffffff'
            },
            align: 'center',
            arrowhead: 2,
            arrowsize: 1,
            arrowwidth: 2,
            arrowcolor: '#636363',
            ax: -200,
            ay: -150,
            bordercolor: '#c7c7c7',
            borderwidth: 2,
            borderpad: 4,
            bgcolor: '#ff7f0e',
            opacity: 0.8
        }, ]
    };
    // Combine the trace and layout
    const data5 = [trace6];

    // Render the chart
    Plotly.newPlot(idDiv, data5, layout);
}
let getAllData = (data) => {
    let high = data.high[data.high.length - 1]
    let adj_Close = data.adj_Close[data.high.length - 1]
    let volume = data.volume[data.high.length - 1]
    let close = data.close[data.high.length - 1]
    document.getElementById('high').innerText = 'High: ' + high
    document.getElementById('adj').innerText = 'Adj_CLose: ' + adj_Close
    document.getElementById('volume').innerText = 'Volume: ' + volume
    document.getElementById('close').innerText = 'Close: ' + close
}
d3.csv("https://raw.githubusercontent.com/Chaudeptrai123456/DataViz/master/BTC-USD.csv", function(raw_data) {
    let data = rowConverter(raw_data)
    drawCandleSimple("simpleCandleChart1", data, '')
        //drawBarChart(data.date, data.close, data.open, 'close', 'open', 'simple_BarChar_Open_Close')
        //drawBarChart(data.date, data.high, data.low, 'high', 'low', 'simple_Bar_Char_High_Low')

    drawtwoline('linetwocolumn2', data, '')
    drawtwoline1('linetwocolumn4', data.date, data.close, data.open, "Close", "Open", "CLose - Open")
    drawtwoline1('linetwocolumn5', data.date, data.high, data.low, "High", "Low", "High - Low")
    getAllData(data)
        // addFeater("feater",data)
})
let array = [...document.querySelectorAll(".control")]
array.forEach(button => {
    button.addEventListener('click', () => {
        document.querySelector(".active").classList.remove("active");
        document.getElementById("page" + button.id).classList.add("active");
    })
})
console.log(data)