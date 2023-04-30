import React from 'react'

const Input = () => {
    return (  
        <div className = "flex flex-col">
            <form>
                <div className="text-white text-center font-mono">Enter link: </div>
                <div className = "flex justify-center">
                    <input type="text" className="w-2/3 bg-white rounded-md border-gray-300 border px-3 py-2 placeholder-gray-400 text-gray-700 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:ring-opacity-50" placeholder="Enter URL here" />
                </div>
            </form>
        </div>
    );
};

export default Input
